use pyo3::prelude::*;

mod cphase_table;
mod small_tables;
mod vop_table;
const SYMMETRIES: usize = 24;
const MEAS_AXES: usize = 3;

/// Graph-state based quantum circuit simulator exposed as the `graphsim` Python module.
#[pymodule]
pub mod graphsim {
    use pyo3::prelude::*;
    use std::{
        collections::{HashMap, HashSet, VecDeque},
        fmt::{self, Debug, Display, Formatter},
        iter::repeat_n,
        ops::{Index, IndexMut, Mul},
    };

    use rand::{
        Rng,
        distr::{Distribution, StandardUniform},
    };

    use crate::{
        cphase_table::CPHASE_TABLE,
        small_tables::{ADJ_TABLE, CONJ_TABLE, DETM_TABLE},
        vop_table::VOP_TABLE,
    };

    /// Index of a node / qubit in the graph.
    pub type NodeIdx = usize;

    /// Result of a single-qubit measurement.
    ///
    /// Exposed to Python as `graphsim.MeasurementResult`.
    #[pyclass(eq, eq_int)]
    #[derive(PartialEq, Eq, Debug, Clone, Copy)]
    pub enum MeasurementResult {
        /// Eigenvalue +1 outcome.
        PlusOne,
        /// Eigenvalue −1 outcome.
        MinusOne,
    }

    impl Display for MeasurementResult {
        fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
            match self {
                MeasurementResult::PlusOne => fmt.write_str("+1"),
                MeasurementResult::MinusOne => fmt.write_str("-1"),
            }
        }
    }

    /// Measurement outcome and the axis that was measured.
    ///
    /// Returned in the values of `peek_measure_set`.
    #[pyclass(frozen, get_all, str)]
    pub struct Outcome {
        result: MeasurementResult,
        axis: Axis,
    }

    impl Display for Outcome {
        fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
            write!(fmt, "({}, {})", self.axis, self.result)
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub(crate) enum Vop {
        IA,
        XA,
        YA,
        ZA,
        IB,
        XB,
        YB,
        ZB,
        IC,
        XC,
        YC,
        ZC,
        ID,
        XD,
        YD,
        ZD,
        IE,
        XE,
        YE,
        ZE,
        IF,
        XF,
        YF,
        ZF,
    }

    #[pyclass(eq, eq_int)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Axis {
        X,
        Y,
        Z,
    }

    impl Display for Axis {
        fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
            match self {
                Axis::X => fmt.write_str("X"),
                Axis::Y => fmt.write_str("Y"),
                Axis::Z => fmt.write_str("Z"),
            }
        }
    }

    impl Distribution<Axis> for StandardUniform {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Axis {
            match rng.random_range(0..3) {
                0 => Axis::X,
                1 => Axis::Y,
                2 => Axis::Z,
                _ => unreachable!("rng generates in the range 0..3"),
            }
        }
    }

    #[derive(Debug)]
    enum Zeta {
        Zero,
        Two,
    }

    impl Distribution<MeasurementResult> for StandardUniform {
        fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> MeasurementResult {
            match rng.random() {
                false => MeasurementResult::PlusOne,
                true => MeasurementResult::MinusOne,
            }
        }
    }

    const X_GATE: Vop = Vop::XA;
    const Y_GATE: Vop = Vop::YA;
    const Z_GATE: Vop = Vop::ZA;
    const H_GATE: Vop = Vop::YC;
    const S_GATE: Vop = Vop::YB;
    const SDAG_GATE: Vop = Vop::XB;

    impl Mul for Vop {
        type Output = Vop;

        fn mul(self, rhs: Self) -> Self::Output {
            VOP_TABLE[self as usize][rhs as usize]
        }
    }

    enum DecompUnit {
        U,
        V,
    }

    impl Vop {
        fn adj(self) -> Self {
            ADJ_TABLE[self as usize]
        }

        fn is_in_z(self) -> bool {
            match self {
                Vop::IA | Vop::ZA | Vop::YB | Vop::XB => true,
                _ => false,
            }
        }

        fn decomp(self) -> &'static [DecompUnit] {
            match self {
                Vop::IA => &[DecompUnit::U, DecompUnit::U, DecompUnit::U, DecompUnit::U],
                Vop::XA => &[DecompUnit::U, DecompUnit::U],
                Vop::YA => &[DecompUnit::U, DecompUnit::U, DecompUnit::V, DecompUnit::V],
                Vop::ZA => &[DecompUnit::V, DecompUnit::V],
                Vop::IB => &[DecompUnit::U, DecompUnit::U, DecompUnit::V],
                Vop::XB => &[DecompUnit::V],
                Vop::YB => &[DecompUnit::V, DecompUnit::V, DecompUnit::V],
                Vop::ZB => &[DecompUnit::V, DecompUnit::U, DecompUnit::U],
                Vop::IC => &[DecompUnit::U, DecompUnit::V, DecompUnit::U],
                Vop::XC => &[
                    DecompUnit::U,
                    DecompUnit::U,
                    DecompUnit::U,
                    DecompUnit::V,
                    DecompUnit::U,
                ],
                Vop::YC => &[
                    DecompUnit::U,
                    DecompUnit::V,
                    DecompUnit::V,
                    DecompUnit::V,
                    DecompUnit::U,
                ],
                Vop::ZC => &[
                    DecompUnit::U,
                    DecompUnit::V,
                    DecompUnit::U,
                    DecompUnit::U,
                    DecompUnit::U,
                ],
                Vop::ID => &[DecompUnit::V, DecompUnit::V, DecompUnit::U],
                Vop::XD => &[DecompUnit::U, DecompUnit::V, DecompUnit::V],
                Vop::YD => &[DecompUnit::U, DecompUnit::U, DecompUnit::U],
                Vop::ZD => &[DecompUnit::U],
                Vop::IE => &[DecompUnit::U, DecompUnit::V, DecompUnit::V, DecompUnit::V],
                Vop::XE => &[DecompUnit::U, DecompUnit::V, DecompUnit::U, DecompUnit::U],
                Vop::YE => &[DecompUnit::U, DecompUnit::V],
                Vop::ZE => &[DecompUnit::U, DecompUnit::U, DecompUnit::U, DecompUnit::V],
                Vop::IF => &[DecompUnit::V, DecompUnit::U, DecompUnit::U, DecompUnit::U],
                Vop::XF => &[DecompUnit::V, DecompUnit::V, DecompUnit::V, DecompUnit::U],
                Vop::YF => &[DecompUnit::V, DecompUnit::U],
                Vop::ZF => &[DecompUnit::U, DecompUnit::U, DecompUnit::V, DecompUnit::U],
            }
        }
    }

    #[derive(Clone)]
    pub struct Node {
        adjacent: HashSet<NodeIdx>,
        vop: Vop,
    }

    impl Debug for Node {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Node")
                .field("adjacent", &self.adjacent)
                .field("vop", &self.vop)
                .field("disp", &self.get_state_str())
                .finish()
        }
    }

    impl Node {
        fn get_state_str(&self) -> &'static str {
            match self.vop {
                Vop::IA | Vop::XA | Vop::YD | Vop::ZD => "+",
                Vop::YA | Vop::ZA | Vop::ID | Vop::XD => "-",
                Vop::IB | Vop::XB | Vop::YE | Vop::ZE => "+i",
                Vop::YB | Vop::ZB | Vop::IE | Vop::XE => "-i",
                Vop::IC | Vop::XC | Vop::YF | Vop::ZF => "1",
                Vop::YC | Vop::ZC | Vop::IF | Vop::XF => "0",
            }
        }
    }

    impl Default for Node {
        fn default() -> Self {
            Self {
                adjacent: HashSet::new(),
                vop: Vop::YC,
            }
        }
    }

    /// Simulator for graph states over a fixed number of qubits.
    ///
    /// Use this class from Python to apply gates and perform measurements.
    #[derive(Clone)]
    #[pyclass]
    pub struct GraphSim {
        nodes: Vec<Node>,
    }

    impl Index<NodeIdx> for GraphSim {
        type Output = Node;

        fn index(&self, index: NodeIdx) -> &Self::Output {
            &self.nodes[index]
        }
    }

    impl IndexMut<NodeIdx> for GraphSim {
        fn index_mut(&mut self, index: NodeIdx) -> &mut Self::Output {
            &mut self.nodes[index]
        }
    }

    impl GraphSim {
        pub fn adj_hist(&self, hm: &mut HashMap<usize, usize>) {
            for node in self.nodes.iter() {
                let l = node.adjacent.len();
                if let Some(val) = hm.get_mut(&l) {
                    *val += 1;
                } else {
                    hm.insert(l, 1);
                }
            }
        }
        // Measurement
        fn measure(&mut self, node: NodeIdx, axis: Axis) -> (MeasurementResult, bool) {
            let zeta = find_zeta(self[node].vop.adj(), axis);
            let basis = &CONJ_TABLE[axis as usize][self[node].vop.adj() as usize];

            let (mut res, deterministic) = match basis {
                Axis::X => self.int_measure_x(node),
                Axis::Y => (self.int_measure_y(node), false),
                Axis::Z => (self.int_measure_z(node), false),
            };

            match zeta {
                Zeta::Two => {
                    res = match res {
                        MeasurementResult::PlusOne => MeasurementResult::MinusOne,
                        MeasurementResult::MinusOne => MeasurementResult::PlusOne,
                    }
                }
                _ => {}
            };

            (res, deterministic)
        }
        fn int_measure_x(&mut self, node: NodeIdx) -> (MeasurementResult, bool) {
            if self[node].adjacent.is_empty() {
                return (MeasurementResult::PlusOne, true);
            }

            let res: MeasurementResult = rand::rng().random();
            let other: NodeIdx = self[node].adjacent.take(&1).unwrap();

            match res {
                MeasurementResult::PlusOne => {
                    self[other].vop = self[other].vop * Vop::ZC;
                    for third in self[node].adjacent.clone().iter() {
                        if *third != other && !self[other].adjacent.contains(third) {
                            self[*third].vop = self[*third].vop * Z_GATE;
                        }
                    }
                }
                MeasurementResult::MinusOne => {
                    self[other].vop = self[other].vop * Vop::XC;
                    self[node].vop = self[node].vop * Vop::ZA;

                    for third in self[other].adjacent.clone().iter() {
                        if *third != other && !self[other].adjacent.contains(third) {
                            self[*third].vop = self[*third].vop * Z_GATE;
                        }
                    }
                }
            }

            let node_nbs = self[node].adjacent.clone();
            let other_nbs = self[other].adjacent.clone();

            let mut procced_edges: HashSet<(NodeIdx, NodeIdx)> = HashSet::new();
            for nval in node_nbs.iter() {
                for oval in other_nbs.iter() {
                    let combined = match nval < oval {
                        true => (nval, oval),
                        false => (oval, nval),
                    };
                    if nval != oval && !procced_edges.contains(&(*combined.0, *combined.1)) {
                        procced_edges.insert((*combined.0, *combined.1));
                        self.toggle_edge(*combined.0, *combined.1);
                    }
                }
            }

            let intersection: Vec<_> = node_nbs.intersection(&other_nbs).into_iter().collect();
            let ilen = intersection.len();
            for i in 0..ilen {
                for j in (i + 1)..ilen {
                    self.toggle_edge(*intersection[i], *intersection[j]);
                }
            }

            for nval in node_nbs {
                if nval != other {
                    self.toggle_edge(other, nval);
                }
            }

            (res, false)
        }
        fn int_measure_y(&mut self, node: NodeIdx) -> MeasurementResult {
            let res = rand::rng().random();

            for &other in self[node].adjacent.clone().iter() {
                match res {
                    MeasurementResult::PlusOne => self[other].vop = self[other].vop * S_GATE,
                    MeasurementResult::MinusOne => self[other].vop = self[other].vop * SDAG_GATE,
                }
            }

            let adj: Vec<_> = self[node].adjacent.clone().into_iter().collect();
            let nlen = adj.len();

            for i in 0..nlen {
                let nval = adj[i];
                for j in i + 1..=nlen {
                    let oval = if j == nlen { node } else { adj[j] };
                    self.toggle_edge(nval, oval);
                }
            }

            match res {
                MeasurementResult::PlusOne => self[node].vop = self[node].vop * S_GATE,
                MeasurementResult::MinusOne => self[node].vop = self[node].vop * SDAG_GATE,
            }

            res
        }
        fn int_measure_z(&mut self, node: NodeIdx) -> MeasurementResult {
            let res = rand::rng().random();

            for &other in self[node].adjacent.clone().iter() {
                self.delete_edge(node, other);
                if res == MeasurementResult::MinusOne {
                    self[other].vop = self[other].vop * Z_GATE;
                }
            }

            match res {
                MeasurementResult::PlusOne => self[node].vop = self[node].vop * H_GATE,
                MeasurementResult::MinusOne => self[node].vop = self[node].vop * X_GATE * H_GATE,
            }

            res
        }
        // Helper functions
        /// remove the local operators non-Z stabilisation by swapping with its neighbours
        ///
        /// Scales as O(?)
        fn remove_vop(&mut self, first: NodeIdx, avoid: NodeIdx) {
            let mut second: NodeIdx = avoid;
            for attempt in &self[first].adjacent {
                if *attempt != avoid {
                    second = *attempt;
                    break;
                }
            }

            for d in self[first].vop.decomp() {
                match d {
                    DecompUnit::U => self.local_comp(first),
                    DecompUnit::V => self.local_comp(second),
                }
            }
        }

        // set = {12,125,123,198,5}
        // index i, j
        // i = 0 -> j = 1..5
        // i = 1 -> j = 2..5
        // i = 2 -> j = 3..5
        // i = 3 -> j = 4..5
        // i = 4 -> j = 5..5
        //
        //
        // for i in set.iter() {
        //   for j in set.iter().skip(i) {
        //     do_something(set[i], set[j])
        //   }
        // }

        /// do a local complementation of a qubit with its surroundings
        ///
        /// Scales as O(?)
        fn local_comp(&mut self, node: NodeIdx) {
            let rself: *mut Self = self as *mut Self;

            for i in self[node].adjacent.iter() {
                for j in self[node].adjacent.iter().skip(*i) {
                    unsafe {
                        let dupself = &mut *rself;
                        dupself.toggle_edge(*i, *j);
                    }
                }
                unsafe {
                    let dupself = &mut *rself;
                    dupself[*i].vop = dupself[*i].vop * S_GATE;
                }
            }
            self[node].vop = self[node].vop * Vop::YD;
        }

        fn toggle_edge(&mut self, na: NodeIdx, nb: NodeIdx) -> bool {
            let a_has_b = self[na].adjacent.remove(&nb);
            let b_has_a = self[nb].adjacent.remove(&na);
            assert_eq!(a_has_b, b_has_a);

            if a_has_b {
                true
            } else {
                self[na].adjacent.insert(nb);
                self[nb].adjacent.insert(na);
                false
            }
        }
        fn delete_edge(&mut self, na: NodeIdx, nb: NodeIdx) {
            self[na].adjacent.remove(&nb);
            self[nb].adjacent.remove(&na);
        }

        fn find_deterministic(&self, node: NodeIdx) -> Option<Axis> {
            if self[node].adjacent.is_empty() {
                Some(DETM_TABLE[self[node].vop.adj() as usize])
            } else {
                None
            }
        }
    }

    #[pymethods]
    impl GraphSim {
        /// Create a new simulator with `nodes` qubits, all initialized in the |0⟩ state.
        #[new]
        pub fn new(qubit_amount: usize) -> GraphSim {
            GraphSim {
                nodes: repeat_n(Node::default(), qubit_amount).collect(),
            }
        }

        /// Apply an X (Pauli-X) gate to the given qubit.
        ///
        /// `node` is the index of the qubit.
        pub fn x(&mut self, qubit: NodeIdx) {
            self[qubit].vop = X_GATE * self[qubit].vop;
        }

        /// Apply a Y (Pauli-Y) gate to the given qubit.
        ///
        /// `node` is the index of the qubit.
        pub fn y(&mut self, qubit: NodeIdx) {
            self[qubit].vop = Y_GATE * self[qubit].vop;
        }

        /// Apply a Z (Pauli-Z) gate to the given qubit.
        ///
        /// `node` is the index of the qubit.
        pub fn z(&mut self, qubit: NodeIdx) {
            self[qubit].vop = Z_GATE * self[qubit].vop;
        }

        /// Apply an H (Hadamard) gate to the given qubit.
        ///
        /// `node` is the index of the qubit.
        pub fn h(&mut self, qubit: NodeIdx) {
            self[qubit].vop = H_GATE * self[qubit].vop;
        }

        /// Apply an S (phase) gate to the given qubit.
        ///
        /// `node` is the index of the qubit.
        pub fn s(&mut self, qubit: NodeIdx) {
            self[qubit].vop = S_GATE * self[qubit].vop;
        }

        /// Apply an S† (inverse phase) gate to the given qubit.
        ///
        /// `node` is the index of the qubit.
        pub fn sdag(&mut self, qubit: NodeIdx) {
            self[qubit].vop = SDAG_GATE * self[qubit].vop;
        }

        /// Apply a controlled-Z (CZ) gate with `control` and `target` qubits.
        pub fn cz(&mut self, control: NodeIdx, target: NodeIdx) {
            // println!(
            //     "performing cnot between {control} and {target}, with adjacent {:#?} and {:#?} respectively",
            //     self[control].adjacent, self[target].adjacent
            // );
            assert_ne!(control, target, "Same control and target not allowed");
            let c_has_non_t = self[control].adjacent.len()
                >= match self[control].adjacent.contains(&target) {
                    true => 2,
                    false => 1,
                };
            let t_has_non_c = self[target].adjacent.len()
                >= match self[target].adjacent.contains(&control) {
                    true => 2,
                    false => 1,
                };

            if c_has_non_t {
                self.remove_vop(control, target);
            }
            if t_has_non_c {
                self.remove_vop(target, control);
            }
            if c_has_non_t && !self[control].vop.is_in_z() {
                self.remove_vop(control, target);
            }

            let cv = self[control].vop;
            let tv = self[target].vop;
            let had_edge = match self[control].adjacent.contains(&target) {
                true => 1,
                false => 0,
            };
            let val = CPHASE_TABLE[had_edge][cv as usize][tv as usize];

            if val.0 {
                self[control].adjacent.insert(target);
                self[target].adjacent.insert(control);
            } else {
                self[control].adjacent.remove(&target);
                self[target].adjacent.remove(&control);
            }
            self[control].vop = val.1;
            self[target].vop = val.2;
        }

        /// Apply a controlled-X (CX) / CNOT gate with `control` and `target`.
        pub fn cx(&mut self, control: NodeIdx, target: NodeIdx) {
            self.h(target);
            self.cz(control, target);
            self.h(target);
        }

        /// Apply an X-controlled X gate (CX in the X basis).
        pub fn xcx(&mut self, control: NodeIdx, target: NodeIdx) {
            self.h(control);
            self.cx(control, target);
            self.h(control);
        }

        /// Apply a Y-controlled X gate (control qubit in the Y basis).
        pub fn ycx(&mut self, control: NodeIdx, target: NodeIdx) {
            self.sdag(control);
            self.xcx(control, target);
            self.s(control);
        }

        /// Apply an X-controlled Z gate (target in X basis).
        pub fn xcz(&mut self, control: NodeIdx, target: NodeIdx) {
            self.cx(target, control);
        }

        /// Apply a Y-controlled Z gate (target in Y basis).
        pub fn ycz(&mut self, control: NodeIdx, target: NodeIdx) {
            self.cy(target, control);
        }

        /// Apply a controlled-Y (CY) gate with `control` and `target`.
        pub fn cy(&mut self, control: NodeIdx, target: NodeIdx) {
            self.sdag(target);
            self.cx(control, target);
            self.s(target);
        }

        /// Apply an X-controlled Y gate (control in X basis).
        pub fn xcy(&mut self, control: NodeIdx, target: NodeIdx) {
            self.ycx(target, control);
        }

        /// Apply a Y-controlled Y gate (both in Y basis).
        pub fn ycy(&mut self, control: NodeIdx, target: NodeIdx) {
            self.sdag(target);
            self.ycx(control, target);
            self.s(target);
        }

        /// Perform a projective measurement of `qubit` in the X basis.
        ///
        /// Returns `MeasurementResult.PlusOne` or `MeasurementResult.MinusOne`.
        pub fn measure_x(&mut self, qubit: NodeIdx) -> MeasurementResult {
            let (res, _) = self.measure(qubit, Axis::X);
            res
        }

        /// Perform a projective measurement of `qubit` in the Y basis.
        ///
        /// Returns `MeasurementResult.PlusOne` or `MeasurementResult.MinusOne`.
        pub fn measure_y(&mut self, qubit: NodeIdx) -> MeasurementResult {
            let (res, _) = self.measure(qubit, Axis::Y);
            res
        }

        /// Perform a projective measurement of `qubit` in the Z basis.
        ///
        /// Returns `MeasurementResult.PlusOne` or `MeasurementResult.MinusOne`.
        pub fn measure_z(&mut self, qubit: NodeIdx) -> MeasurementResult {
            let (res, _) = self.measure(qubit, Axis::Z);
            res
        }

        /// Return the set of qubits that are entangled with `qubit`.
        ///
        /// This follows adjacency in the underlying graph.
        pub fn get_entangled_group(&self, qubit: NodeIdx) -> HashSet<NodeIdx> {
            let mut queue = VecDeque::new();
            let mut part = HashSet::new();
            queue.push_back(qubit);
            part.insert(qubit);
            while let Some(val) = queue.pop_front() {
                for adj in &self[val].adjacent {
                    if !part.contains(adj) {
                        queue.push_back(*adj);
                        part.insert(*adj);
                    }
                }
            }

            part
        }

        /// Simulate measurements on a set of `qubits` without modifying the real state.
        ///
        /// Returns a map from qubit index to `Outcome` (result and axis used).
        pub fn peek_measure_set(
            &self,
            qubits: HashSet<NodeIdx>,
        ) -> std::collections::HashMap<NodeIdx, Outcome> {
            let mut changeset = self.clone();
            qubits
                .iter()
                .map(|&idx| {
                    let axis = if let Some(deterministic) = changeset.find_deterministic(idx) {
                        deterministic
                    } else {
                        rand::rng().random()
                    };

                    let (result, _) = changeset.measure(idx, axis);

                    (idx, Outcome { result, axis })
                })
                .collect()
        }
    }

    fn find_zeta(vop: Vop, axis: Axis) -> Zeta {
        let rvop = (vop as usize) & 0b11;

        match (
            (rvop == 0 || rvop == (axis as usize + 1)),
            vop >= Vop::IB && vop < Vop::IE,
        ) {
            (true, true) | (false, false) => Zeta::Two,
            (true, false) | (false, true) => Zeta::Zero,
        }
    }
}
