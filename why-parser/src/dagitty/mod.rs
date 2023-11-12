#![allow(missing_docs)]

use std::{collections::HashMap, marker::PhantomData};

use pest::{error::Error, iterators::Pair, Parser};
use pest_derive::Parser;
use why_data::{
    graph::{
        dagitty::{EdgeInfo, EdgeType, NodeInfo, VertexType},
        CausalGraph, Graph, NodeIndex, UnGraph,
    },
    types::Point,
};

// Parser
#[derive(Debug, Parser)]
#[grammar = "dagitty/dagitty.pest"]
pub struct DagittyParser;

impl DagittyParser {
    fn parse_edge_rhs(
        pair: Pair<'_, Rule>,
        mut builder: CausalGraphBuilder<NodeInfo, EdgeInfo>,
        left_node_id: &str,
        pos: Option<(f64, f64)>,
    ) -> Result<CausalGraphBuilder<NodeInfo, EdgeInfo>, Error<Rule>> {
        debug_assert!(
            Rule::edge_rhs == pair.as_rule(),
            "Input must be an edge_rhs rule"
        );
        let mut inners = pair.into_inner();
        let edgeop = inners.next().unwrap().as_str();
        let node_id = inners
            .next()
            .map(|p| {
                let mut inners = p.into_inner();
                inners.next().unwrap().as_str()
            })
            .unwrap();

        match edgeop {
            "@->" | "->" => {
                let edge = EdgeInfo::new("", pos.map(|p| Point::new(p.0, p.1)), EdgeType::Directed);
                builder = builder.add_edge(left_node_id, node_id, edge);
            }
            "<-@" | "<-" => {
                let edge = EdgeInfo::new("", pos.map(|p| Point::new(p.0, p.1)), EdgeType::Directed);
                builder = builder.add_edge(node_id, left_node_id, edge);
            }
            "--@" | "--" | "<->" | "@-@" | "@--" => {
                let edge =
                    EdgeInfo::new("", pos.map(|p| Point::new(p.0, p.1)), EdgeType::Undirected);
                builder = builder.add_edge(left_node_id, node_id, edge);
            }
            _ => unreachable!(),
        }

        if inners.peek().is_some() {
            builder = Self::parse_edge_rhs(inners.next().unwrap(), builder, node_id, pos)?;
        }

        Ok(builder)
    }

    fn parse_edge(
        pair: Pair<'_, Rule>,
        mut builder: CausalGraphBuilder<NodeInfo, EdgeInfo>,
    ) -> Result<CausalGraphBuilder<NodeInfo, EdgeInfo>, Error<Rule>> {
        let mut pos = None;
        let mut inners = pair.into_inner();
        let node_id = inners
            .next()
            .map(|p| {
                let mut inners = p.into_inner();
                inners.next().unwrap().as_str()
            })
            .unwrap();
        let edge_rhs = inners.next().unwrap();
        let attrs = inners.next().map(|p| AttrList::parse(p).unwrap());
        if let Some(attrs) = attrs {
            for alist in attrs.elems {
                for attr in alist.elems {
                    if let ("pos", position) = attr {
                        let (x, y) = position.split_at(position.find(',').unwrap_or(0));
                        pos = Some((
                            x.parse::<f64>().unwrap_or(0.0),
                            y.parse::<f64>().unwrap_or(0.0),
                        ));
                    }
                }
            }
        }

        builder = Self::parse_edge_rhs(edge_rhs, builder, node_id, pos)?;
        Ok(builder)
    }

    fn parse_node(
        pair: Pair<'_, Rule>,
        mut builder: CausalGraphBuilder<NodeInfo, EdgeInfo>,
    ) -> Result<CausalGraphBuilder<NodeInfo, EdgeInfo>, Error<Rule>> {
        let mut vertex_type = VertexType::None;
        let mut pos = (0.0, 0.0);
        let mut inners = pair.into_inner();
        let node_id = inners
            .next()
            .map(|p| {
                let mut inners = p.into_inner();
                inners.next().unwrap().as_str()
            })
            .unwrap();
        let attrs = inners.next().map(|p| AttrList::parse(p).unwrap());
        if let Some(attrs) = attrs {
            for alist in attrs.elems {
                for attr in alist.elems {
                    match attr {
                        ("adjusted", _) | ("a", _) => vertex_type = VertexType::Adjusted,
                        ("source", _) | ("exposure", _) | ("e", _) => {
                            vertex_type = VertexType::Exposure
                        }
                        ("outcome", _) | ("target", _) | ("o", _) => {
                            vertex_type = VertexType::Outcome
                        }
                        ("selected", _) | ("s", _) => vertex_type = VertexType::Selected,
                        ("latent", _) | ("l", _) | ("unobserved", _) | ("u", _) => {
                            vertex_type = VertexType::Unobserved
                        }
                        ("pos", position) => {
                            let (x, y) = position.split_at(position.find(',').unwrap_or(0));
                            pos = (
                                x.parse::<f64>().unwrap_or(0.0),
                                y.parse::<f64>().unwrap_or(0.0),
                            );
                        }
                        (_, _) => (),
                    }
                }
            }
        }
        let node_info = NodeInfo::new(node_id, pos.0, pos.1, vertex_type);
        builder = builder.add_node(node_info, node_id);

        Ok(builder)
    }

    fn parse_stmt(
        pair: Pair<'_, Rule>,
        mut builder: CausalGraphBuilder<NodeInfo, EdgeInfo>,
    ) -> Result<CausalGraphBuilder<NodeInfo, EdgeInfo>, Error<Rule>> {
        let inner = pair.into_inner().next().unwrap();
        builder = match inner.as_rule() {
            Rule::node_stmt => Self::parse_node(inner, builder)?,
            Rule::edge_stmt => Self::parse_edge(inner, builder)?,
            Rule::global_option => todo!(),
            _ => unreachable!(),
        };

        Ok(builder)
    }
    fn parse_stmts(
        pair: Pair<'_, Rule>,
        mut builder: CausalGraphBuilder<NodeInfo, EdgeInfo>,
    ) -> Result<CausalGraphBuilder<NodeInfo, EdgeInfo>, Error<Rule>> {
        let mut inner = pair.into_inner();
        match inner.next() {
            None => {}
            Some(stmt) => {
                let tail = inner.next().unwrap();
                builder = Self::parse_stmt(stmt, builder)?;
                builder = Self::parse_stmts(tail, builder)?;
            }
        }

        Ok(builder)
    }

    /// Parse dagitty format to create a casual graph.
    pub fn parse_str(content: &str) -> Result<CausalGraph<NodeInfo, EdgeInfo>, Error<Rule>> {
        let mut builder = CausalGraphBuilder::<NodeInfo, EdgeInfo>::new();
        let mut parser = DagittyParser::parse(Rule::dagitty_graph, content)?;
        let mut dagitty_g = parser.next().unwrap().into_inner();
        let mut _strict = false;
        let mut pair = dagitty_g.next().unwrap();
        if let Rule::STRICT = pair.as_rule() {
            _strict = true;
            pair = dagitty_g.next().unwrap();
        }

        builder = match pair.as_str() {
            "digraph" | "dag" => builder.dag(),
            "graph" | "mag" | "pdag" | "pag" => builder.graph(),
            &_ => unreachable!("Unknown graph string"),
        };

        builder = Self::parse_stmts(pair, builder)?;

        Ok(builder.build())
    }
}

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone)]
struct AttrList<'a, A = (&'a str, &'a str)> {
    /// The list of `AList`s.
    pub(crate) elems: Vec<AList<'a, A>>,
}

impl<'a> AttrList<'a> {
    fn parse(p: Pair<'a, Rule>) -> Result<Self, ()> {
        debug_assert!(
            Rule::attr_list == p.as_rule(),
            "Input must be an attr_list rule"
        );
        let mut v: Vec<AList<'a>> = Vec::new();
        let mut inners = p.into_inner();
        let alist = AList::parse(inners.next().unwrap()).unwrap();
        let mut tail = inners
            .next()
            .map(|p| {
                AttrList::parse(p)
                    .map(|alist| alist.elems)
                    .unwrap_or(Vec::new())
            })
            .unwrap_or(Vec::new());
        v.push(alist);
        v.append(&mut tail);

        Ok(AttrList { elems: v })
    }
}

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Clone)]
struct AList<'a, A = (&'a str, &'a str)> {
    /// The attributes in the list.
    pub(crate) elems: Vec<A>,
    _p: PhantomData<&'a ()>,
}

impl<'a> AList<'a> {
    fn parse(p: Pair<'a, Rule>) -> Result<Self, ()> {
        debug_assert!(Rule::a_list == p.as_rule(), "Input must be an a_list rule");
        let mut v = Vec::new();
        let mut inners = p.into_inner();
        let pair = inners.next().unwrap();

        let (id1, id2) = if Rule::id_eq == pair.as_rule() {
            let mut ideq_pair = pair.into_inner();
            let id1 = ideq_pair.next().unwrap().as_str();
            let id2 = ideq_pair.next().unwrap().as_str();
            (id1, id2)
        } else {
            (pair.as_str(), "")
        };
        let mut tail = inners
            .next()
            .map(|p| {
                AList::parse(p)
                    .map(|alist| alist.elems)
                    .unwrap_or(Vec::new())
            })
            .unwrap_or(Vec::new());
        v.push((id1, id2));
        v.append(&mut tail);

        Ok(AList {
            elems: v,
            _p: PhantomData,
        })
    }
}

struct CausalGraphBuilder<N, E> {
    graph: Option<CausalGraph<N, E>>,
    node_map: HashMap<String, NodeIndex>,
}

impl<N, E> CausalGraphBuilder<N, E> {
    fn new() -> CausalGraphBuilder<N, E> {
        Self {
            graph: None,
            node_map: HashMap::new(),
        }
    }

    fn dag(mut self) -> CausalGraphBuilder<N, E> {
        let g = Graph::<N, E>::new();
        self.graph = Some(CausalGraph::Dag(g));
        self
    }

    fn graph(mut self) -> CausalGraphBuilder<N, E> {
        let g = UnGraph::<N, E>::new_undirected();
        self.graph = Some(CausalGraph::Ungraph(g));
        self
    }

    fn add_node(mut self, n: N, id: &str) -> CausalGraphBuilder<N, E> {
        if self.graph.is_some() {
            let g = self.graph.as_mut().unwrap();
            self.node_map.insert(id.into(), g.add_node(n));
        }
        self
    }

    fn add_edge(mut self, left_node: &str, right_node: &str, edge: E) -> CausalGraphBuilder<N, E> {
        let left_id = self.node_map.get(left_node);
        let right_id = self.node_map.get(right_node);

        if self.graph.is_some() && left_id.is_some() && right_id.is_some() {
            let g = self.graph.as_mut().unwrap();
            g.add_edge(*left_id.unwrap(), *right_id.unwrap(), edge);
        }

        self
    }

    fn build(self) -> CausalGraph<N, E> {
        self.graph.unwrap()
    }
}

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

    #[test]
    fn test_parse_a_list() {
        let mut pairs =
            DagittyParser::parse(Rule::a_list, r#"selected,pos="-2.200,-1.520""#).unwrap();
        let alist = AList::parse(pairs.next().unwrap()).unwrap();

        assert_eq!(
            [("selected", ""), ("pos", "\"-2.200,-1.520\"")],
            &alist.elems[..]
        );
    }

    #[test]
    fn test_parse_attr_list() {
        let mut pairs =
            DagittyParser::parse(Rule::attr_list, r#"[selected,pos="-2.200,-1.520"]"#).unwrap();
        let attr_list = AttrList::parse(pairs.next().unwrap()).unwrap();

        assert_eq!(
            [("selected", ""), ("pos", "\"-2.200,-1.520\"")],
            &(attr_list.elems[0]).elems[..]
        );
    }
}
