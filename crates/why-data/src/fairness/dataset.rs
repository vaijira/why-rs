//! A minimal categorical table, with a CSV reader.

use std::collections::{HashMap, HashSet};

use super::FairnessError;

/// One column, interned: `levels[codes[i]]` is the value in row `i`.
#[derive(Debug, Clone)]
pub(crate) struct Column {
    pub(crate) levels: Vec<String>,
    pub(crate) codes: Vec<u32>,
}

/// A table of string-valued columns, with optional per-row weights.
///
/// Every value is kept as a categorical level. The outcome column is parsed
/// as a number only when an estimator asks for it, so one table can serve
/// both a binary `Y` and a real-valued prediction `Ŷ`.
///
/// ```
/// use why_data::fairness::Dataset;
///
/// let data = Dataset::from_csv("x,y\na,1\nb,0\n").unwrap();
/// assert_eq!(data.len(), 2);
/// assert_eq!(data.levels("x").unwrap(), ["a", "b"]);
/// ```
#[derive(Debug, Clone)]
pub struct Dataset {
    names: Vec<String>,
    columns: Vec<Column>,
    weights: Option<Vec<f64>>,
    len: usize,
}

impl Dataset {
    /// Builds a table from a header and records of string-like fields.
    pub fn from_records<H, R, F, S>(header: H, records: R) -> Result<Self, FairnessError>
    where
        H: IntoIterator,
        H::Item: Into<String>,
        R: IntoIterator<Item = F>,
        F: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let names: Vec<String> = header.into_iter().map(Into::into).collect();
        let mut seen = HashSet::new();
        for name in &names {
            if !seen.insert(name.as_str()) {
                return Err(FairnessError::DuplicateColumn(name.clone()));
            }
        }

        let mut lookups: Vec<HashMap<String, u32>> = vec![HashMap::new(); names.len()];
        let mut columns: Vec<Column> = names
            .iter()
            .map(|_| Column {
                levels: Vec::new(),
                codes: Vec::new(),
            })
            .collect();
        let mut len = 0;

        for (row, record) in records.into_iter().enumerate() {
            let fields: Vec<S> = record.into_iter().collect();
            if fields.len() != names.len() {
                return Err(FairnessError::RaggedRow {
                    row: row + 1,
                    expected: names.len(),
                    found: fields.len(),
                });
            }
            for ((field, lookup), column) in fields.iter().zip(&mut lookups).zip(&mut columns) {
                let value = field.as_ref();
                let code = match lookup.get(value) {
                    Some(&code) => code,
                    None => {
                        let code = column.levels.len() as u32;
                        column.levels.push(value.to_string());
                        lookup.insert(value.to_string(), code);
                        code
                    }
                };
                column.codes.push(code);
            }
            len += 1;
        }

        Ok(Self {
            names,
            columns,
            weights: None,
            len,
        })
    }

    /// Parses comma-separated text whose first record is the header.
    ///
    /// Supports RFC 4180 quoting (`"a,b"`, `"say ""hi"""`), CRLF line endings
    /// and blank lines. Unquoted fields are trimmed.
    pub fn from_csv(text: &str) -> Result<Self, FairnessError> {
        let mut records = parse_csv(text)?.into_iter();
        let header = records.next().ok_or(FairnessError::Csv {
            line: 1,
            message: "no header".into(),
        })?;
        Self::from_records(header, records)
    }

    /// Attaches one non-negative weight per row, e.g. frequency counts or
    /// survey weights. Every estimate then uses weighted proportions.
    pub fn with_weights(mut self, weights: Vec<f64>) -> Result<Self, FairnessError> {
        if weights.len() != self.len {
            return Err(FairnessError::InvalidWeights(format!(
                "{} weights for {} rows",
                weights.len(),
                self.len
            )));
        }
        if let Some(bad) = weights.iter().find(|w| !w.is_finite() || **w < 0.0) {
            return Err(FairnessError::InvalidWeights(format!(
                "weight {bad} is negative or not finite"
            )));
        }
        self.weights = Some(weights);
        Ok(self)
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the table has no rows.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Column names, in header order.
    pub fn column_names(&self) -> &[String] {
        &self.names
    }

    /// Distinct values of a column, in order of first appearance.
    pub fn levels(&self, name: &str) -> Option<&[String]> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| self.columns[i].levels.as_slice())
    }

    /// Returns a copy with one more column. Errors if the name is taken or
    /// the number of values differs from the number of rows.
    pub fn with_column<I>(mut self, name: impl Into<String>, values: I) -> Result<Self, FairnessError>
    where
        I: IntoIterator,
        I::Item: Into<String>,
    {
        let name = name.into();
        if self.names.contains(&name) {
            return Err(FairnessError::DuplicateColumn(name));
        }
        let mut lookup: HashMap<String, u32> = HashMap::new();
        let mut column = Column {
            levels: Vec::new(),
            codes: Vec::with_capacity(self.len),
        };
        for value in values {
            let value: String = value.into();
            let code = match lookup.get(&value) {
                Some(&code) => code,
                None => {
                    let code = column.levels.len() as u32;
                    column.levels.push(value.clone());
                    lookup.insert(value, code);
                    code
                }
            };
            column.codes.push(code);
        }
        if column.codes.len() != self.len {
            return Err(FairnessError::InvalidInput(format!(
                "column `{name}` has {} values for {} rows",
                column.codes.len(),
                self.len
            )));
        }
        self.names.push(name);
        self.columns.push(column);
        Ok(self)
    }

    /// Per-row weights, if any were attached.
    pub fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }

    /// The value in row `row` of column `name`.
    pub fn value(&self, name: &str, row: usize) -> Option<&str> {
        let col = self.column(name).ok()?;
        col.codes
            .get(row)
            .map(|&code| col.levels[code as usize].as_str())
    }

    /// Serialises the table as CSV (weights are not included). Fields that
    /// contain commas, quotes, newlines or surrounding spaces are quoted.
    pub fn to_csv(&self) -> String {
        fn field(out: &mut String, v: &str) {
            let needs = v.contains([',', '"', '\n', '\r']) || v.trim() != v;
            if needs {
                out.push('"');
                out.push_str(&v.replace('"', "\"\""));
                out.push('"');
            } else {
                out.push_str(v);
            }
        }
        let mut out = String::new();
        for (i, n) in self.names.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            field(&mut out, n);
        }
        out.push('\n');
        for row in 0..self.len {
            for (i, c) in self.columns.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                field(&mut out, &c.levels[c.codes[row] as usize]);
            }
            out.push('\n');
        }
        out
    }

    pub(crate) fn column(&self, name: &str) -> Result<&Column, FairnessError> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| &self.columns[i])
            .ok_or_else(|| FairnessError::UnknownColumn(name.to_string()))
    }

    pub(crate) fn weight(&self, row: usize) -> f64 {
        self.weights.as_ref().map_or(1.0, |w| w[row])
    }
}

fn parse_csv(text: &str) -> Result<Vec<Vec<String>>, FairnessError> {
    let mut records = Vec::new();
    let mut record: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut quoted = false;
    let mut line = 1;
    let mut chars = text.chars().peekable();

    let finish = |field: &mut String, quoted: &mut bool| {
        let value = if *quoted {
            std::mem::take(field)
        } else {
            let trimmed = field.trim().to_string();
            field.clear();
            trimmed
        };
        *quoted = false;
        value
    };

    while let Some(c) = chars.next() {
        if in_quotes {
            match c {
                '"' if chars.peek() == Some(&'"') => {
                    chars.next();
                    field.push('"');
                }
                '"' => in_quotes = false,
                '\n' => {
                    line += 1;
                    field.push(c);
                }
                _ => field.push(c),
            }
            continue;
        }
        match c {
            '"' if !quoted && field.trim().is_empty() => {
                field.clear();
                in_quotes = true;
                quoted = true;
            }
            ',' => record.push(finish(&mut field, &mut quoted)),
            '\r' => {}
            '\n' => {
                let blank = record.is_empty() && !quoted && field.trim().is_empty();
                if !blank {
                    record.push(finish(&mut field, &mut quoted));
                    records.push(std::mem::take(&mut record));
                }
                field.clear();
                line += 1;
            }
            _ if quoted => {
                if !c.is_whitespace() {
                    return Err(FairnessError::Csv {
                        line,
                        message: format!("unexpected `{c}` after closing quote"),
                    });
                }
            }
            _ => field.push(c),
        }
    }
    if in_quotes {
        return Err(FairnessError::Csv {
            line,
            message: "unterminated quoted field".into(),
        });
    }
    if !(record.is_empty() && !quoted && field.trim().is_empty()) {
        record.push(finish(&mut field, &mut quoted));
        records.push(record);
    }
    Ok(records)
}
