use pest_derive::Parser;

/// Dagitty format parser.
#[derive(Debug, Parser)]
#[grammar = "dagitty.pest"]
pub struct DagittyParser;

#[cfg(test)]
mod tests {
    use super::*;
    use pest::Parser;

    const BASE_DAG_STR: &str = r#"dag {
A [selected,pos="-2.200,-1.520"]
B [pos="1.400,-1.460"]
D [outcome,pos="1.400,1.621"]
E [exposure,pos="-2.200,1.597"]
Z [adjusted,pos="-0.300,-0.082"]
A -> E
A -> Z [pos="-0.791,-1.045"]
B -> D
B -> Z [pos="0.680,-0.496"]
E -> D
}"#;

    #[test]
    fn test_parser() {
        let mut parser = DagittyParser::parse(Rule::dagitty_graph, BASE_DAG_STR).unwrap();
        let mut dagitty_g = parser.next().unwrap().into_inner();
        let mut strict = false;
        let mut name = None;
        let mut pair = dagitty_g.next().unwrap();
        if let Rule::STRICT = pair.as_rule() {
            strict = true;
            pair = dagitty_g.next().unwrap();
        }
        assert_eq!(false, strict);
        assert_eq!(Rule::GRAPHTYPE, pair.as_rule());
        assert_eq!("dag", pair.as_str());
        if let Rule::IDENTIFIER = pair.as_rule() {
            name = Some(pair.as_str());
        }
        assert_eq!(None, name);
    }
}
