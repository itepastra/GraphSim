use pyo3::prelude::*;

/// Graph-state based quantum circuit simulator exposed as the `graphsim` Python module.
#[pymodule]
mod graphsim {
    use pyo3::prelude::*;
    use std::{
        collections::{HashSet, VecDeque},
        fmt::{self, Debug},
        iter::repeat_n,
        ops::{Index, IndexMut, Mul},
    };

    use rand::{
        Rng,
        distr::{Distribution, StandardUniform},
    };

    /// Index of a node / qubit in the graph.
    pub type NodeIdx = usize;

    /// Result of a single-qubit measurement.
    ///
    /// Exposed to Python as `graphsim.MeasurementResult`.
    #[pyclass]
    #[derive(PartialEq, Eq, Debug)]
    pub enum MeasurementResult {
        /// Eigenvalue +1 outcome.
        PlusOne,
        /// Eigenvalue −1 outcome.
        MinusOne,
    }

    /// Measurement outcome and the axis that was measured.
    ///
    /// Returned in the values of `peek_measure_set`.
    #[pyclass]
    pub struct Outcome {
        result: MeasurementResult,
        axis: Axis,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    enum Vop {
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

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Axis {
        X,
        Y,
        Z,
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

    const VOP_TABLE: [[Vop; SYMMETRIES]; SYMMETRIES] = [
        [
            Vop::IA,
            Vop::XA,
            Vop::YA,
            Vop::ZA,
            Vop::IB,
            Vop::XB,
            Vop::YB,
            Vop::ZB,
            Vop::IC,
            Vop::XC,
            Vop::YC,
            Vop::ZC,
            Vop::ID,
            Vop::XD,
            Vop::YD,
            Vop::ZD,
            Vop::IE,
            Vop::XE,
            Vop::YE,
            Vop::ZE,
            Vop::IF,
            Vop::XF,
            Vop::YF,
            Vop::ZF,
        ],
        [
            Vop::XA,
            Vop::IA,
            Vop::ZA,
            Vop::YA,
            Vop::YB,
            Vop::ZB,
            Vop::IB,
            Vop::XB,
            Vop::ZC,
            Vop::YC,
            Vop::XC,
            Vop::IC,
            Vop::XD,
            Vop::ID,
            Vop::ZD,
            Vop::YD,
            Vop::ZE,
            Vop::YE,
            Vop::XE,
            Vop::IE,
            Vop::YF,
            Vop::ZF,
            Vop::IF,
            Vop::XF,
        ],
        [
            Vop::YA,
            Vop::ZA,
            Vop::IA,
            Vop::XA,
            Vop::XB,
            Vop::IB,
            Vop::ZB,
            Vop::YB,
            Vop::YC,
            Vop::ZC,
            Vop::IC,
            Vop::XC,
            Vop::ZD,
            Vop::YD,
            Vop::XD,
            Vop::ID,
            Vop::XE,
            Vop::IE,
            Vop::ZE,
            Vop::YE,
            Vop::ZF,
            Vop::YF,
            Vop::XF,
            Vop::IF,
        ],
        [
            Vop::ZA,
            Vop::YA,
            Vop::XA,
            Vop::IA,
            Vop::ZB,
            Vop::YB,
            Vop::XB,
            Vop::IB,
            Vop::XC,
            Vop::IC,
            Vop::ZC,
            Vop::YC,
            Vop::YD,
            Vop::ZD,
            Vop::ID,
            Vop::XD,
            Vop::YE,
            Vop::ZE,
            Vop::IE,
            Vop::XE,
            Vop::XF,
            Vop::IF,
            Vop::ZF,
            Vop::YF,
        ],
        [
            Vop::IB,
            Vop::XB,
            Vop::YB,
            Vop::ZB,
            Vop::IA,
            Vop::XA,
            Vop::YA,
            Vop::ZA,
            Vop::IF,
            Vop::XF,
            Vop::YF,
            Vop::ZF,
            Vop::IE,
            Vop::XE,
            Vop::YE,
            Vop::ZE,
            Vop::ID,
            Vop::XD,
            Vop::YD,
            Vop::ZD,
            Vop::IC,
            Vop::XC,
            Vop::YC,
            Vop::ZC,
        ],
        [
            Vop::XB,
            Vop::IB,
            Vop::ZB,
            Vop::YB,
            Vop::YA,
            Vop::ZA,
            Vop::IA,
            Vop::XA,
            Vop::ZF,
            Vop::YF,
            Vop::XF,
            Vop::IF,
            Vop::XE,
            Vop::IE,
            Vop::ZE,
            Vop::YE,
            Vop::ZD,
            Vop::YD,
            Vop::XD,
            Vop::ID,
            Vop::YC,
            Vop::ZC,
            Vop::IC,
            Vop::XC,
        ],
        [
            Vop::YB,
            Vop::ZB,
            Vop::IB,
            Vop::XB,
            Vop::XA,
            Vop::IA,
            Vop::ZA,
            Vop::YA,
            Vop::YF,
            Vop::ZF,
            Vop::IF,
            Vop::XF,
            Vop::ZE,
            Vop::YE,
            Vop::XE,
            Vop::IE,
            Vop::XD,
            Vop::ID,
            Vop::ZD,
            Vop::YD,
            Vop::ZC,
            Vop::YC,
            Vop::XC,
            Vop::IC,
        ],
        [
            Vop::ZB,
            Vop::YB,
            Vop::XB,
            Vop::IB,
            Vop::ZA,
            Vop::YA,
            Vop::XA,
            Vop::IA,
            Vop::XF,
            Vop::IF,
            Vop::ZF,
            Vop::YF,
            Vop::YE,
            Vop::ZE,
            Vop::IE,
            Vop::XE,
            Vop::YD,
            Vop::ZD,
            Vop::ID,
            Vop::XD,
            Vop::XC,
            Vop::IC,
            Vop::ZC,
            Vop::YC,
        ],
        [
            Vop::IC,
            Vop::XC,
            Vop::YC,
            Vop::ZC,
            Vop::IE,
            Vop::XE,
            Vop::YE,
            Vop::ZE,
            Vop::IA,
            Vop::XA,
            Vop::YA,
            Vop::ZA,
            Vop::IF,
            Vop::XF,
            Vop::YF,
            Vop::ZF,
            Vop::IB,
            Vop::XB,
            Vop::YB,
            Vop::ZB,
            Vop::ID,
            Vop::XD,
            Vop::YD,
            Vop::ZD,
        ],
        [
            Vop::XC,
            Vop::IC,
            Vop::ZC,
            Vop::YC,
            Vop::YE,
            Vop::ZE,
            Vop::IE,
            Vop::XE,
            Vop::ZA,
            Vop::YA,
            Vop::XA,
            Vop::IA,
            Vop::XF,
            Vop::IF,
            Vop::ZF,
            Vop::YF,
            Vop::ZB,
            Vop::YB,
            Vop::XB,
            Vop::IB,
            Vop::YD,
            Vop::ZD,
            Vop::ID,
            Vop::XD,
        ],
        [
            Vop::YC,
            Vop::ZC,
            Vop::IC,
            Vop::XC,
            Vop::XE,
            Vop::IE,
            Vop::ZE,
            Vop::YE,
            Vop::YA,
            Vop::ZA,
            Vop::IA,
            Vop::XA,
            Vop::ZF,
            Vop::YF,
            Vop::XF,
            Vop::IF,
            Vop::XB,
            Vop::IB,
            Vop::ZB,
            Vop::YB,
            Vop::ZD,
            Vop::YD,
            Vop::XD,
            Vop::ID,
        ],
        [
            Vop::ZC,
            Vop::YC,
            Vop::XC,
            Vop::IC,
            Vop::ZE,
            Vop::YE,
            Vop::XE,
            Vop::IE,
            Vop::XA,
            Vop::IA,
            Vop::ZA,
            Vop::YA,
            Vop::YF,
            Vop::ZF,
            Vop::IF,
            Vop::XF,
            Vop::YB,
            Vop::ZB,
            Vop::IB,
            Vop::XB,
            Vop::XD,
            Vop::ID,
            Vop::ZD,
            Vop::YD,
        ],
        [
            Vop::ID,
            Vop::XD,
            Vop::YD,
            Vop::ZD,
            Vop::IF,
            Vop::XF,
            Vop::YF,
            Vop::ZF,
            Vop::IE,
            Vop::XE,
            Vop::YE,
            Vop::ZE,
            Vop::IA,
            Vop::XA,
            Vop::YA,
            Vop::ZA,
            Vop::IC,
            Vop::XC,
            Vop::YC,
            Vop::ZC,
            Vop::IB,
            Vop::XB,
            Vop::YB,
            Vop::ZB,
        ],
        [
            Vop::XD,
            Vop::ID,
            Vop::ZD,
            Vop::YD,
            Vop::YF,
            Vop::ZF,
            Vop::IF,
            Vop::XF,
            Vop::ZE,
            Vop::YE,
            Vop::XE,
            Vop::IE,
            Vop::XA,
            Vop::IA,
            Vop::ZA,
            Vop::YA,
            Vop::ZC,
            Vop::YC,
            Vop::XC,
            Vop::IC,
            Vop::YB,
            Vop::ZB,
            Vop::IB,
            Vop::XB,
        ],
        [
            Vop::YD,
            Vop::ZD,
            Vop::ID,
            Vop::XD,
            Vop::XF,
            Vop::IF,
            Vop::ZF,
            Vop::YF,
            Vop::YE,
            Vop::ZE,
            Vop::IE,
            Vop::XE,
            Vop::ZA,
            Vop::YA,
            Vop::XA,
            Vop::IA,
            Vop::XC,
            Vop::IC,
            Vop::ZC,
            Vop::YC,
            Vop::ZB,
            Vop::YB,
            Vop::XB,
            Vop::IB,
        ],
        [
            Vop::ZD,
            Vop::YD,
            Vop::XD,
            Vop::ID,
            Vop::ZF,
            Vop::YF,
            Vop::XF,
            Vop::IF,
            Vop::XE,
            Vop::IE,
            Vop::ZE,
            Vop::YE,
            Vop::YA,
            Vop::ZA,
            Vop::IA,
            Vop::XA,
            Vop::YC,
            Vop::ZC,
            Vop::IC,
            Vop::XC,
            Vop::XB,
            Vop::IB,
            Vop::ZB,
            Vop::YB,
        ],
        [
            Vop::IE,
            Vop::XE,
            Vop::YE,
            Vop::ZE,
            Vop::IC,
            Vop::XC,
            Vop::YC,
            Vop::ZC,
            Vop::ID,
            Vop::XD,
            Vop::YD,
            Vop::ZD,
            Vop::IB,
            Vop::XB,
            Vop::YB,
            Vop::ZB,
            Vop::IF,
            Vop::XF,
            Vop::YF,
            Vop::ZF,
            Vop::IA,
            Vop::XA,
            Vop::YA,
            Vop::ZA,
        ],
        [
            Vop::XE,
            Vop::IE,
            Vop::ZE,
            Vop::YE,
            Vop::YC,
            Vop::ZC,
            Vop::IC,
            Vop::XC,
            Vop::ZD,
            Vop::YD,
            Vop::XD,
            Vop::ID,
            Vop::XB,
            Vop::IB,
            Vop::ZB,
            Vop::YB,
            Vop::ZF,
            Vop::YF,
            Vop::XF,
            Vop::IF,
            Vop::YA,
            Vop::ZA,
            Vop::IA,
            Vop::XA,
        ],
        [
            Vop::YE,
            Vop::ZE,
            Vop::IE,
            Vop::XE,
            Vop::XC,
            Vop::IC,
            Vop::ZC,
            Vop::YC,
            Vop::YD,
            Vop::ZD,
            Vop::ID,
            Vop::XD,
            Vop::ZB,
            Vop::YB,
            Vop::XB,
            Vop::IB,
            Vop::XF,
            Vop::IF,
            Vop::ZF,
            Vop::YF,
            Vop::ZA,
            Vop::YA,
            Vop::XA,
            Vop::IA,
        ],
        [
            Vop::ZE,
            Vop::YE,
            Vop::XE,
            Vop::IE,
            Vop::ZC,
            Vop::YC,
            Vop::XC,
            Vop::IC,
            Vop::XD,
            Vop::ID,
            Vop::ZD,
            Vop::YD,
            Vop::YB,
            Vop::ZB,
            Vop::IB,
            Vop::XB,
            Vop::YF,
            Vop::ZF,
            Vop::IF,
            Vop::XF,
            Vop::XA,
            Vop::IA,
            Vop::ZA,
            Vop::YA,
        ],
        [
            Vop::IF,
            Vop::XF,
            Vop::YF,
            Vop::ZF,
            Vop::ID,
            Vop::XD,
            Vop::YD,
            Vop::ZD,
            Vop::IB,
            Vop::XB,
            Vop::YB,
            Vop::ZB,
            Vop::IC,
            Vop::XC,
            Vop::YC,
            Vop::ZC,
            Vop::IA,
            Vop::XA,
            Vop::YA,
            Vop::ZA,
            Vop::IE,
            Vop::XE,
            Vop::YE,
            Vop::ZE,
        ],
        [
            Vop::XF,
            Vop::IF,
            Vop::ZF,
            Vop::YF,
            Vop::YD,
            Vop::ZD,
            Vop::ID,
            Vop::XD,
            Vop::ZB,
            Vop::YB,
            Vop::XB,
            Vop::IB,
            Vop::XC,
            Vop::IC,
            Vop::ZC,
            Vop::YC,
            Vop::ZA,
            Vop::YA,
            Vop::XA,
            Vop::IA,
            Vop::YE,
            Vop::ZE,
            Vop::IE,
            Vop::XE,
        ],
        [
            Vop::YF,
            Vop::ZF,
            Vop::IF,
            Vop::XF,
            Vop::XD,
            Vop::ID,
            Vop::ZD,
            Vop::YD,
            Vop::YB,
            Vop::ZB,
            Vop::IB,
            Vop::XB,
            Vop::ZC,
            Vop::YC,
            Vop::XC,
            Vop::IC,
            Vop::XA,
            Vop::IA,
            Vop::ZA,
            Vop::YA,
            Vop::ZE,
            Vop::YE,
            Vop::XE,
            Vop::IE,
        ],
        [
            Vop::ZF,
            Vop::YF,
            Vop::XF,
            Vop::IF,
            Vop::ZD,
            Vop::YD,
            Vop::XD,
            Vop::ID,
            Vop::XB,
            Vop::IB,
            Vop::ZB,
            Vop::YB,
            Vop::YC,
            Vop::ZC,
            Vop::IC,
            Vop::XC,
            Vop::YA,
            Vop::ZA,
            Vop::IA,
            Vop::XA,
            Vop::XE,
            Vop::IE,
            Vop::ZE,
            Vop::YE,
        ],
    ];

    const ADJ_TABLE: [Vop; SYMMETRIES] = [
        Vop::IA,
        Vop::XA,
        Vop::YA,
        Vop::ZA,
        Vop::IB,
        Vop::YB,
        Vop::XB,
        Vop::ZB,
        Vop::IC,
        Vop::ZC,
        Vop::YC,
        Vop::XC,
        Vop::ID,
        Vop::XD,
        Vop::ZD,
        Vop::YD,
        Vop::IF,
        Vop::YF,
        Vop::ZF,
        Vop::XF,
        Vop::IE,
        Vop::ZE,
        Vop::XE,
        Vop::YE,
    ];

    const DETM_TABLE: [Axis; SYMMETRIES] = [
        Axis::X,
        Axis::X,
        Axis::X,
        Axis::X,
        Axis::Y,
        Axis::Y,
        Axis::Y,
        Axis::Y,
        Axis::Z,
        Axis::Z,
        Axis::Z,
        Axis::Z,
        Axis::X,
        Axis::X,
        Axis::X,
        Axis::X,
        Axis::Z,
        Axis::Z,
        Axis::Z,
        Axis::Z,
        Axis::Y,
        Axis::Y,
        Axis::Y,
        Axis::Y,
    ];

    const CONJ_TABLE: [[Axis; SYMMETRIES]; MEAS_AXES] = [
        [
            Axis::X,
            Axis::X,
            Axis::X,
            Axis::X,
            Axis::Y,
            Axis::Y,
            Axis::Y,
            Axis::Y,
            Axis::Z,
            Axis::Z,
            Axis::Z,
            Axis::Z,
            Axis::X,
            Axis::X,
            Axis::X,
            Axis::X,
            Axis::Y,
            Axis::Y,
            Axis::Y,
            Axis::Y,
            Axis::Z,
            Axis::Z,
            Axis::Z,
            Axis::Z,
        ],
        [
            Axis::Y,
            Axis::Y,
            Axis::Y,
            Axis::Y,
            Axis::X,
            Axis::X,
            Axis::X,
            Axis::X,
            Axis::Y,
            Axis::Y,
            Axis::Y,
            Axis::Y,
            Axis::Z,
            Axis::Z,
            Axis::Z,
            Axis::Z,
            Axis::Z,
            Axis::Z,
            Axis::Z,
            Axis::Z,
            Axis::X,
            Axis::X,
            Axis::X,
            Axis::X,
        ],
        [
            Axis::Z,
            Axis::Z,
            Axis::Z,
            Axis::Z,
            Axis::Z,
            Axis::Z,
            Axis::Z,
            Axis::Z,
            Axis::X,
            Axis::X,
            Axis::X,
            Axis::X,
            Axis::Y,
            Axis::Y,
            Axis::Y,
            Axis::Y,
            Axis::X,
            Axis::X,
            Axis::X,
            Axis::X,
            Axis::Y,
            Axis::Y,
            Axis::Y,
            Axis::Y,
        ],
    ];

    const CPHASE_TABLE: [[[(bool, Vop, Vop); SYMMETRIES]; SYMMETRIES]; 2] = [
        [
            [
                (true, Vop::IA, Vop::IA),
                (true, Vop::IA, Vop::IA),
                (true, Vop::IA, Vop::ZA),
                (true, Vop::IA, Vop::ZA),
                (true, Vop::IA, Vop::XB),
                (true, Vop::IA, Vop::XB),
                (true, Vop::IA, Vop::YB),
                (true, Vop::IA, Vop::YB),
                (false, Vop::ZA, Vop::IC),
                (false, Vop::ZA, Vop::IC),
                (false, Vop::IA, Vop::YC),
                (false, Vop::IA, Vop::YC),
                (true, Vop::IA, Vop::ZA),
                (true, Vop::IA, Vop::ZA),
                (true, Vop::IA, Vop::IA),
                (true, Vop::IA, Vop::IA),
                (true, Vop::IA, Vop::YB),
                (true, Vop::IA, Vop::YB),
                (true, Vop::IA, Vop::XB),
                (true, Vop::IA, Vop::XB),
                (false, Vop::IA, Vop::YC),
                (false, Vop::IA, Vop::YC),
                (false, Vop::ZA, Vop::IC),
                (false, Vop::ZA, Vop::IC),
            ],
            [
                (true, Vop::IA, Vop::IA),
                (true, Vop::IA, Vop::IA),
                (true, Vop::IA, Vop::ZA),
                (true, Vop::IA, Vop::ZA),
                (true, Vop::IA, Vop::XB),
                (true, Vop::IA, Vop::XB),
                (true, Vop::IA, Vop::YB),
                (true, Vop::IA, Vop::YB),
                (false, Vop::YA, Vop::IC),
                (false, Vop::YA, Vop::IC),
                (false, Vop::IA, Vop::YC),
                (false, Vop::IA, Vop::YC),
                (true, Vop::IA, Vop::ZA),
                (true, Vop::IA, Vop::ZA),
                (true, Vop::IA, Vop::IA),
                (true, Vop::IA, Vop::IA),
                (true, Vop::IA, Vop::YB),
                (true, Vop::IA, Vop::YB),
                (true, Vop::IA, Vop::XB),
                (true, Vop::IA, Vop::XB),
                (false, Vop::IA, Vop::YC),
                (false, Vop::IA, Vop::YC),
                (false, Vop::YA, Vop::IC),
                (false, Vop::YA, Vop::IC),
            ],
            [
                (true, Vop::YA, Vop::ZA),
                (true, Vop::IA, Vop::XA),
                (true, Vop::IA, Vop::YA),
                (true, Vop::YA, Vop::IA),
                (true, Vop::IA, Vop::IB),
                (true, Vop::YA, Vop::YB),
                (true, Vop::YA, Vop::XB),
                (true, Vop::IA, Vop::ZB),
                (false, Vop::IA, Vop::IC),
                (false, Vop::IA, Vop::IC),
                (false, Vop::YA, Vop::YC),
                (false, Vop::YA, Vop::YC),
                (true, Vop::IA, Vop::YA),
                (true, Vop::IA, Vop::YA),
                (true, Vop::IA, Vop::XA),
                (true, Vop::IA, Vop::XA),
                (true, Vop::IA, Vop::ZB),
                (true, Vop::IA, Vop::ZB),
                (true, Vop::IA, Vop::IB),
                (true, Vop::IA, Vop::IB),
                (false, Vop::YA, Vop::YC),
                (false, Vop::YA, Vop::YC),
                (false, Vop::IA, Vop::IC),
                (false, Vop::IA, Vop::IC),
            ],
            [
                (true, Vop::ZA, Vop::IA),
                (true, Vop::IA, Vop::XA),
                (true, Vop::IA, Vop::YA),
                (true, Vop::ZA, Vop::ZA),
                (true, Vop::IA, Vop::IB),
                (true, Vop::ZA, Vop::XB),
                (true, Vop::ZA, Vop::YB),
                (true, Vop::IA, Vop::ZB),
                (false, Vop::IA, Vop::IC),
                (false, Vop::IA, Vop::IC),
                (false, Vop::ZA, Vop::YC),
                (false, Vop::ZA, Vop::YC),
                (true, Vop::IA, Vop::YA),
                (true, Vop::IA, Vop::YA),
                (true, Vop::IA, Vop::XA),
                (true, Vop::IA, Vop::XA),
                (true, Vop::IA, Vop::ZB),
                (true, Vop::IA, Vop::ZB),
                (true, Vop::IA, Vop::IB),
                (true, Vop::IA, Vop::IB),
                (false, Vop::ZA, Vop::YC),
                (false, Vop::ZA, Vop::YC),
                (false, Vop::IA, Vop::IC),
                (false, Vop::IA, Vop::IC),
            ],
            [
                (true, Vop::IB, Vop::ZA),
                (true, Vop::IB, Vop::ZA),
                (true, Vop::IB, Vop::IA),
                (true, Vop::IB, Vop::IA),
                (true, Vop::IB, Vop::YB),
                (true, Vop::IB, Vop::YB),
                (true, Vop::IB, Vop::XB),
                (true, Vop::IB, Vop::XB),
                (false, Vop::YB, Vop::IC),
                (false, Vop::YB, Vop::IC),
                (false, Vop::IB, Vop::YC),
                (false, Vop::IB, Vop::YC),
                (true, Vop::IB, Vop::IA),
                (true, Vop::IB, Vop::IA),
                (true, Vop::IB, Vop::ZA),
                (true, Vop::IB, Vop::ZA),
                (true, Vop::IB, Vop::XB),
                (true, Vop::IB, Vop::XB),
                (true, Vop::IB, Vop::YB),
                (true, Vop::IB, Vop::YB),
                (false, Vop::IB, Vop::YC),
                (false, Vop::IB, Vop::YC),
                (false, Vop::YB, Vop::IC),
                (false, Vop::YB, Vop::IC),
            ],
            [
                (true, Vop::XB, Vop::IA),
                (true, Vop::XB, Vop::IA),
                (true, Vop::XB, Vop::ZA),
                (true, Vop::XB, Vop::ZA),
                (true, Vop::XB, Vop::XB),
                (true, Vop::XB, Vop::XB),
                (true, Vop::XB, Vop::YB),
                (true, Vop::XB, Vop::YB),
                (false, Vop::YB, Vop::IC),
                (false, Vop::YB, Vop::IC),
                (false, Vop::XB, Vop::YC),
                (false, Vop::XB, Vop::YC),
                (true, Vop::XB, Vop::ZA),
                (true, Vop::XB, Vop::ZA),
                (true, Vop::XB, Vop::IA),
                (true, Vop::XB, Vop::IA),
                (true, Vop::XB, Vop::YB),
                (true, Vop::XB, Vop::YB),
                (true, Vop::XB, Vop::XB),
                (true, Vop::XB, Vop::XB),
                (false, Vop::XB, Vop::YC),
                (false, Vop::XB, Vop::YC),
                (false, Vop::YB, Vop::IC),
                (false, Vop::YB, Vop::IC),
            ],
            [
                (true, Vop::YB, Vop::IA),
                (true, Vop::XB, Vop::XA),
                (true, Vop::XB, Vop::YA),
                (true, Vop::YB, Vop::ZA),
                (true, Vop::XB, Vop::IB),
                (true, Vop::YB, Vop::XB),
                (true, Vop::YB, Vop::YB),
                (true, Vop::XB, Vop::ZB),
                (false, Vop::XB, Vop::IC),
                (false, Vop::XB, Vop::IC),
                (false, Vop::YB, Vop::YC),
                (false, Vop::YB, Vop::YC),
                (true, Vop::XB, Vop::YA),
                (true, Vop::XB, Vop::YA),
                (true, Vop::XB, Vop::XA),
                (true, Vop::XB, Vop::XA),
                (true, Vop::XB, Vop::ZB),
                (true, Vop::XB, Vop::ZB),
                (true, Vop::XB, Vop::IB),
                (true, Vop::XB, Vop::IB),
                (false, Vop::YB, Vop::YC),
                (false, Vop::YB, Vop::YC),
                (false, Vop::XB, Vop::IC),
                (false, Vop::XB, Vop::IC),
            ],
            [
                (true, Vop::YB, Vop::IA),
                (true, Vop::IB, Vop::YA),
                (true, Vop::IB, Vop::XA),
                (true, Vop::YB, Vop::ZA),
                (true, Vop::IB, Vop::ZB),
                (true, Vop::YB, Vop::XB),
                (true, Vop::YB, Vop::YB),
                (true, Vop::IB, Vop::IB),
                (false, Vop::IB, Vop::IC),
                (false, Vop::IB, Vop::IC),
                (false, Vop::YB, Vop::YC),
                (false, Vop::YB, Vop::YC),
                (true, Vop::IB, Vop::XA),
                (true, Vop::IB, Vop::XA),
                (true, Vop::IB, Vop::YA),
                (true, Vop::IB, Vop::YA),
                (true, Vop::IB, Vop::IB),
                (true, Vop::IB, Vop::IB),
                (true, Vop::IB, Vop::ZB),
                (true, Vop::IB, Vop::ZB),
                (false, Vop::YB, Vop::YC),
                (false, Vop::YB, Vop::YC),
                (false, Vop::IB, Vop::IC),
                (false, Vop::IB, Vop::IC),
            ],
            [
                (false, Vop::IC, Vop::ZA),
                (false, Vop::IC, Vop::YA),
                (false, Vop::IC, Vop::IA),
                (false, Vop::IC, Vop::IA),
                (false, Vop::IC, Vop::YB),
                (false, Vop::IC, Vop::YB),
                (false, Vop::IC, Vop::XB),
                (false, Vop::IC, Vop::IB),
                (false, Vop::IC, Vop::IC),
                (false, Vop::IC, Vop::IC),
                (false, Vop::IC, Vop::YC),
                (false, Vop::IC, Vop::YC),
                (false, Vop::IC, Vop::IA),
                (false, Vop::IC, Vop::IA),
                (false, Vop::IC, Vop::YA),
                (false, Vop::IC, Vop::YA),
                (false, Vop::IC, Vop::IB),
                (false, Vop::IC, Vop::IB),
                (false, Vop::IC, Vop::YB),
                (false, Vop::IC, Vop::YB),
                (false, Vop::IC, Vop::YC),
                (false, Vop::IC, Vop::YC),
                (false, Vop::IC, Vop::IC),
                (false, Vop::IC, Vop::IC),
            ],
            [
                (false, Vop::IC, Vop::ZA),
                (false, Vop::IC, Vop::YA),
                (false, Vop::IC, Vop::IA),
                (false, Vop::IC, Vop::IA),
                (false, Vop::IC, Vop::YB),
                (false, Vop::IC, Vop::YB),
                (false, Vop::IC, Vop::XB),
                (false, Vop::IC, Vop::IB),
                (false, Vop::IC, Vop::IC),
                (false, Vop::IC, Vop::IC),
                (false, Vop::IC, Vop::YC),
                (false, Vop::IC, Vop::YC),
                (false, Vop::IC, Vop::IA),
                (false, Vop::IC, Vop::IA),
                (false, Vop::IC, Vop::YA),
                (false, Vop::IC, Vop::YA),
                (false, Vop::IC, Vop::IB),
                (false, Vop::IC, Vop::IB),
                (false, Vop::IC, Vop::YB),
                (false, Vop::IC, Vop::YB),
                (false, Vop::IC, Vop::YC),
                (false, Vop::IC, Vop::YC),
                (false, Vop::IC, Vop::IC),
                (false, Vop::IC, Vop::IC),
            ],
            [
                (false, Vop::YC, Vop::IA),
                (false, Vop::YC, Vop::IA),
                (false, Vop::YC, Vop::YA),
                (false, Vop::YC, Vop::ZA),
                (false, Vop::YC, Vop::IB),
                (false, Vop::YC, Vop::XB),
                (false, Vop::YC, Vop::YB),
                (false, Vop::YC, Vop::YB),
                (false, Vop::YC, Vop::IC),
                (false, Vop::YC, Vop::IC),
                (false, Vop::YC, Vop::YC),
                (false, Vop::YC, Vop::YC),
                (false, Vop::YC, Vop::YA),
                (false, Vop::YC, Vop::YA),
                (false, Vop::YC, Vop::IA),
                (false, Vop::YC, Vop::IA),
                (false, Vop::YC, Vop::YB),
                (false, Vop::YC, Vop::YB),
                (false, Vop::YC, Vop::IB),
                (false, Vop::YC, Vop::IB),
                (false, Vop::YC, Vop::YC),
                (false, Vop::YC, Vop::YC),
                (false, Vop::YC, Vop::IC),
                (false, Vop::YC, Vop::IC),
            ],
            [
                (false, Vop::YC, Vop::IA),
                (false, Vop::YC, Vop::IA),
                (false, Vop::YC, Vop::YA),
                (false, Vop::YC, Vop::ZA),
                (false, Vop::YC, Vop::IB),
                (false, Vop::YC, Vop::XB),
                (false, Vop::YC, Vop::YB),
                (false, Vop::YC, Vop::YB),
                (false, Vop::YC, Vop::IC),
                (false, Vop::YC, Vop::IC),
                (false, Vop::YC, Vop::YC),
                (false, Vop::YC, Vop::YC),
                (false, Vop::YC, Vop::YA),
                (false, Vop::YC, Vop::YA),
                (false, Vop::YC, Vop::IA),
                (false, Vop::YC, Vop::IA),
                (false, Vop::YC, Vop::YB),
                (false, Vop::YC, Vop::YB),
                (false, Vop::YC, Vop::IB),
                (false, Vop::YC, Vop::IB),
                (false, Vop::YC, Vop::YC),
                (false, Vop::YC, Vop::YC),
                (false, Vop::YC, Vop::IC),
                (false, Vop::YC, Vop::IC),
            ],
            [
                (true, Vop::YA, Vop::ZA),
                (true, Vop::IA, Vop::XA),
                (true, Vop::IA, Vop::YA),
                (true, Vop::YA, Vop::IA),
                (true, Vop::IA, Vop::IB),
                (true, Vop::YA, Vop::YB),
                (true, Vop::YA, Vop::XB),
                (true, Vop::IA, Vop::ZB),
                (false, Vop::IA, Vop::IC),
                (false, Vop::IA, Vop::IC),
                (false, Vop::YA, Vop::YC),
                (false, Vop::YA, Vop::YC),
                (true, Vop::IA, Vop::YA),
                (true, Vop::IA, Vop::YA),
                (true, Vop::IA, Vop::XA),
                (true, Vop::IA, Vop::XA),
                (true, Vop::IA, Vop::ZB),
                (true, Vop::IA, Vop::ZB),
                (true, Vop::IA, Vop::IB),
                (true, Vop::IA, Vop::IB),
                (false, Vop::YA, Vop::YC),
                (false, Vop::YA, Vop::YC),
                (false, Vop::IA, Vop::IC),
                (false, Vop::IA, Vop::IC),
            ],
            [
                (true, Vop::YA, Vop::ZA),
                (true, Vop::IA, Vop::XA),
                (true, Vop::IA, Vop::YA),
                (true, Vop::YA, Vop::IA),
                (true, Vop::IA, Vop::IB),
                (true, Vop::YA, Vop::YB),
                (true, Vop::YA, Vop::XB),
                (true, Vop::IA, Vop::ZB),
                (false, Vop::IA, Vop::IC),
                (false, Vop::IA, Vop::IC),
                (false, Vop::YA, Vop::YC),
                (false, Vop::YA, Vop::YC),
                (true, Vop::IA, Vop::YA),
                (true, Vop::IA, Vop::YA),
                (true, Vop::IA, Vop::XA),
                (true, Vop::IA, Vop::XA),
                (true, Vop::IA, Vop::ZB),
                (true, Vop::IA, Vop::ZB),
                (true, Vop::IA, Vop::IB),
                (true, Vop::IA, Vop::IB),
                (false, Vop::YA, Vop::YC),
                (false, Vop::YA, Vop::YC),
                (false, Vop::IA, Vop::IC),
                (false, Vop::IA, Vop::IC),
            ],
            [
                (true, Vop::IA, Vop::IA),
                (true, Vop::IA, Vop::IA),
                (true, Vop::IA, Vop::ZA),
                (true, Vop::IA, Vop::ZA),
                (true, Vop::IA, Vop::XB),
                (true, Vop::IA, Vop::XB),
                (true, Vop::IA, Vop::YB),
                (true, Vop::IA, Vop::YB),
                (false, Vop::YA, Vop::IC),
                (false, Vop::YA, Vop::IC),
                (false, Vop::IA, Vop::YC),
                (false, Vop::IA, Vop::YC),
                (true, Vop::IA, Vop::ZA),
                (true, Vop::IA, Vop::ZA),
                (true, Vop::IA, Vop::IA),
                (true, Vop::IA, Vop::IA),
                (true, Vop::IA, Vop::YB),
                (true, Vop::IA, Vop::YB),
                (true, Vop::IA, Vop::XB),
                (true, Vop::IA, Vop::XB),
                (false, Vop::IA, Vop::YC),
                (false, Vop::IA, Vop::YC),
                (false, Vop::YA, Vop::IC),
                (false, Vop::YA, Vop::IC),
            ],
            [
                (true, Vop::IA, Vop::IA),
                (true, Vop::IA, Vop::IA),
                (true, Vop::IA, Vop::ZA),
                (true, Vop::IA, Vop::ZA),
                (true, Vop::IA, Vop::XB),
                (true, Vop::IA, Vop::XB),
                (true, Vop::IA, Vop::YB),
                (true, Vop::IA, Vop::YB),
                (false, Vop::YA, Vop::IC),
                (false, Vop::YA, Vop::IC),
                (false, Vop::IA, Vop::YC),
                (false, Vop::IA, Vop::YC),
                (true, Vop::IA, Vop::ZA),
                (true, Vop::IA, Vop::ZA),
                (true, Vop::IA, Vop::IA),
                (true, Vop::IA, Vop::IA),
                (true, Vop::IA, Vop::YB),
                (true, Vop::IA, Vop::YB),
                (true, Vop::IA, Vop::XB),
                (true, Vop::IA, Vop::XB),
                (false, Vop::IA, Vop::YC),
                (false, Vop::IA, Vop::YC),
                (false, Vop::YA, Vop::IC),
                (false, Vop::YA, Vop::IC),
            ],
            [
                (true, Vop::YB, Vop::IA),
                (true, Vop::IB, Vop::YA),
                (true, Vop::IB, Vop::XA),
                (true, Vop::YB, Vop::ZA),
                (true, Vop::IB, Vop::ZB),
                (true, Vop::YB, Vop::XB),
                (true, Vop::YB, Vop::YB),
                (true, Vop::IB, Vop::IB),
                (false, Vop::IB, Vop::IC),
                (false, Vop::IB, Vop::IC),
                (false, Vop::YB, Vop::YC),
                (false, Vop::YB, Vop::YC),
                (true, Vop::IB, Vop::XA),
                (true, Vop::IB, Vop::XA),
                (true, Vop::IB, Vop::YA),
                (true, Vop::IB, Vop::YA),
                (true, Vop::IB, Vop::IB),
                (true, Vop::IB, Vop::IB),
                (true, Vop::IB, Vop::ZB),
                (true, Vop::IB, Vop::ZB),
                (false, Vop::YB, Vop::YC),
                (false, Vop::YB, Vop::YC),
                (false, Vop::IB, Vop::IC),
                (false, Vop::IB, Vop::IC),
            ],
            [
                (true, Vop::YB, Vop::IA),
                (true, Vop::IB, Vop::YA),
                (true, Vop::IB, Vop::XA),
                (true, Vop::YB, Vop::ZA),
                (true, Vop::IB, Vop::ZB),
                (true, Vop::YB, Vop::XB),
                (true, Vop::YB, Vop::YB),
                (true, Vop::IB, Vop::IB),
                (false, Vop::IB, Vop::IC),
                (false, Vop::IB, Vop::IC),
                (false, Vop::YB, Vop::YC),
                (false, Vop::YB, Vop::YC),
                (true, Vop::IB, Vop::XA),
                (true, Vop::IB, Vop::XA),
                (true, Vop::IB, Vop::YA),
                (true, Vop::IB, Vop::YA),
                (true, Vop::IB, Vop::IB),
                (true, Vop::IB, Vop::IB),
                (true, Vop::IB, Vop::ZB),
                (true, Vop::IB, Vop::ZB),
                (false, Vop::YB, Vop::YC),
                (false, Vop::YB, Vop::YC),
                (false, Vop::IB, Vop::IC),
                (false, Vop::IB, Vop::IC),
            ],
            [
                (true, Vop::IB, Vop::ZA),
                (true, Vop::IB, Vop::ZA),
                (true, Vop::IB, Vop::IA),
                (true, Vop::IB, Vop::IA),
                (true, Vop::IB, Vop::YB),
                (true, Vop::IB, Vop::YB),
                (true, Vop::IB, Vop::XB),
                (true, Vop::IB, Vop::XB),
                (false, Vop::YB, Vop::IC),
                (false, Vop::YB, Vop::IC),
                (false, Vop::IB, Vop::YC),
                (false, Vop::IB, Vop::YC),
                (true, Vop::IB, Vop::IA),
                (true, Vop::IB, Vop::IA),
                (true, Vop::IB, Vop::ZA),
                (true, Vop::IB, Vop::ZA),
                (true, Vop::IB, Vop::XB),
                (true, Vop::IB, Vop::XB),
                (true, Vop::IB, Vop::YB),
                (true, Vop::IB, Vop::YB),
                (false, Vop::IB, Vop::YC),
                (false, Vop::IB, Vop::YC),
                (false, Vop::YB, Vop::IC),
                (false, Vop::YB, Vop::IC),
            ],
            [
                (true, Vop::IB, Vop::ZA),
                (true, Vop::IB, Vop::ZA),
                (true, Vop::IB, Vop::IA),
                (true, Vop::IB, Vop::IA),
                (true, Vop::IB, Vop::YB),
                (true, Vop::IB, Vop::YB),
                (true, Vop::IB, Vop::XB),
                (true, Vop::IB, Vop::XB),
                (false, Vop::YB, Vop::IC),
                (false, Vop::YB, Vop::IC),
                (false, Vop::IB, Vop::YC),
                (false, Vop::IB, Vop::YC),
                (true, Vop::IB, Vop::IA),
                (true, Vop::IB, Vop::IA),
                (true, Vop::IB, Vop::ZA),
                (true, Vop::IB, Vop::ZA),
                (true, Vop::IB, Vop::XB),
                (true, Vop::IB, Vop::XB),
                (true, Vop::IB, Vop::YB),
                (true, Vop::IB, Vop::YB),
                (false, Vop::IB, Vop::YC),
                (false, Vop::IB, Vop::YC),
                (false, Vop::YB, Vop::IC),
                (false, Vop::YB, Vop::IC),
            ],
            [
                (false, Vop::YC, Vop::IA),
                (false, Vop::YC, Vop::IA),
                (false, Vop::YC, Vop::YA),
                (false, Vop::YC, Vop::ZA),
                (false, Vop::YC, Vop::IB),
                (false, Vop::YC, Vop::XB),
                (false, Vop::YC, Vop::YB),
                (false, Vop::YC, Vop::YB),
                (false, Vop::YC, Vop::IC),
                (false, Vop::YC, Vop::IC),
                (false, Vop::YC, Vop::YC),
                (false, Vop::YC, Vop::YC),
                (false, Vop::YC, Vop::YA),
                (false, Vop::YC, Vop::YA),
                (false, Vop::YC, Vop::IA),
                (false, Vop::YC, Vop::IA),
                (false, Vop::YC, Vop::YB),
                (false, Vop::YC, Vop::YB),
                (false, Vop::YC, Vop::IB),
                (false, Vop::YC, Vop::IB),
                (false, Vop::YC, Vop::YC),
                (false, Vop::YC, Vop::YC),
                (false, Vop::YC, Vop::IC),
                (false, Vop::YC, Vop::IC),
            ],
            [
                (false, Vop::YC, Vop::IA),
                (false, Vop::YC, Vop::IA),
                (false, Vop::YC, Vop::YA),
                (false, Vop::YC, Vop::ZA),
                (false, Vop::YC, Vop::IB),
                (false, Vop::YC, Vop::XB),
                (false, Vop::YC, Vop::YB),
                (false, Vop::YC, Vop::YB),
                (false, Vop::YC, Vop::IC),
                (false, Vop::YC, Vop::IC),
                (false, Vop::YC, Vop::YC),
                (false, Vop::YC, Vop::YC),
                (false, Vop::YC, Vop::YA),
                (false, Vop::YC, Vop::YA),
                (false, Vop::YC, Vop::IA),
                (false, Vop::YC, Vop::IA),
                (false, Vop::YC, Vop::YB),
                (false, Vop::YC, Vop::YB),
                (false, Vop::YC, Vop::IB),
                (false, Vop::YC, Vop::IB),
                (false, Vop::YC, Vop::YC),
                (false, Vop::YC, Vop::YC),
                (false, Vop::YC, Vop::IC),
                (false, Vop::YC, Vop::IC),
            ],
            [
                (false, Vop::IC, Vop::ZA),
                (false, Vop::IC, Vop::YA),
                (false, Vop::IC, Vop::IA),
                (false, Vop::IC, Vop::IA),
                (false, Vop::IC, Vop::YB),
                (false, Vop::IC, Vop::YB),
                (false, Vop::IC, Vop::XB),
                (false, Vop::IC, Vop::IB),
                (false, Vop::IC, Vop::IC),
                (false, Vop::IC, Vop::IC),
                (false, Vop::IC, Vop::YC),
                (false, Vop::IC, Vop::YC),
                (false, Vop::IC, Vop::IA),
                (false, Vop::IC, Vop::IA),
                (false, Vop::IC, Vop::YA),
                (false, Vop::IC, Vop::YA),
                (false, Vop::IC, Vop::IB),
                (false, Vop::IC, Vop::IB),
                (false, Vop::IC, Vop::YB),
                (false, Vop::IC, Vop::YB),
                (false, Vop::IC, Vop::YC),
                (false, Vop::IC, Vop::YC),
                (false, Vop::IC, Vop::IC),
                (false, Vop::IC, Vop::IC),
            ],
            [
                (false, Vop::IC, Vop::ZA),
                (false, Vop::IC, Vop::YA),
                (false, Vop::IC, Vop::IA),
                (false, Vop::IC, Vop::IA),
                (false, Vop::IC, Vop::YB),
                (false, Vop::IC, Vop::YB),
                (false, Vop::IC, Vop::XB),
                (false, Vop::IC, Vop::IB),
                (false, Vop::IC, Vop::IC),
                (false, Vop::IC, Vop::IC),
                (false, Vop::IC, Vop::YC),
                (false, Vop::IC, Vop::YC),
                (false, Vop::IC, Vop::IA),
                (false, Vop::IC, Vop::IA),
                (false, Vop::IC, Vop::YA),
                (false, Vop::IC, Vop::YA),
                (false, Vop::IC, Vop::IB),
                (false, Vop::IC, Vop::IB),
                (false, Vop::IC, Vop::YB),
                (false, Vop::IC, Vop::YB),
                (false, Vop::IC, Vop::YC),
                (false, Vop::IC, Vop::YC),
                (false, Vop::IC, Vop::IC),
                (false, Vop::IC, Vop::IC),
            ],
        ],
        [
            [
                (false, Vop::IA, Vop::IA),
                (false, Vop::ZA, Vop::IA),
                (false, Vop::ZA, Vop::YA),
                (false, Vop::IA, Vop::ZA),
                (false, Vop::ZA, Vop::IB),
                (false, Vop::IA, Vop::XB),
                (false, Vop::IA, Vop::YB),
                (false, Vop::ZA, Vop::YB),
                (true, Vop::XB, Vop::ZF),
                (true, Vop::XB, Vop::YF),
                (true, Vop::XB, Vop::XF),
                (true, Vop::XB, Vop::IF),
                (false, Vop::XB, Vop::YA),
                (false, Vop::YB, Vop::YA),
                (false, Vop::XB, Vop::IA),
                (false, Vop::YB, Vop::IA),
                (false, Vop::YB, Vop::YB),
                (false, Vop::XB, Vop::YB),
                (false, Vop::YB, Vop::IB),
                (false, Vop::XB, Vop::IB),
                (true, Vop::XB, Vop::YC),
                (true, Vop::XB, Vop::ZC),
                (true, Vop::XB, Vop::IC),
                (true, Vop::XB, Vop::XC),
            ],
            [
                (false, Vop::IA, Vop::ZA),
                (false, Vop::YA, Vop::YA),
                (false, Vop::YA, Vop::IA),
                (false, Vop::IA, Vop::IA),
                (false, Vop::YA, Vop::YB),
                (false, Vop::IA, Vop::YB),
                (false, Vop::IA, Vop::XB),
                (false, Vop::YA, Vop::IB),
                (true, Vop::IB, Vop::ZF),
                (true, Vop::IB, Vop::YF),
                (true, Vop::IB, Vop::XF),
                (true, Vop::IB, Vop::IF),
                (false, Vop::YB, Vop::IA),
                (false, Vop::IB, Vop::IA),
                (false, Vop::YB, Vop::YA),
                (false, Vop::IB, Vop::YA),
                (false, Vop::IB, Vop::IB),
                (false, Vop::YB, Vop::IB),
                (false, Vop::IB, Vop::YB),
                (false, Vop::YB, Vop::YB),
                (true, Vop::IB, Vop::YC),
                (true, Vop::IB, Vop::ZC),
                (true, Vop::IB, Vop::IC),
                (true, Vop::IB, Vop::XC),
            ],
            [
                (false, Vop::YA, Vop::ZA),
                (false, Vop::IA, Vop::YA),
                (false, Vop::IA, Vop::IA),
                (false, Vop::YA, Vop::IA),
                (false, Vop::IA, Vop::YB),
                (false, Vop::YA, Vop::YB),
                (false, Vop::YA, Vop::XB),
                (false, Vop::IA, Vop::IB),
                (true, Vop::IB, Vop::YF),
                (true, Vop::IB, Vop::ZF),
                (true, Vop::IB, Vop::IF),
                (true, Vop::IB, Vop::XF),
                (false, Vop::IB, Vop::IA),
                (false, Vop::YB, Vop::IA),
                (false, Vop::IB, Vop::YA),
                (false, Vop::YB, Vop::YA),
                (false, Vop::YB, Vop::IB),
                (false, Vop::IB, Vop::IB),
                (false, Vop::YB, Vop::YB),
                (false, Vop::IB, Vop::YB),
                (true, Vop::IB, Vop::ZC),
                (true, Vop::IB, Vop::YC),
                (true, Vop::IB, Vop::XC),
                (true, Vop::IB, Vop::IC),
            ],
            [
                (false, Vop::ZA, Vop::IA),
                (false, Vop::IA, Vop::IA),
                (false, Vop::IA, Vop::YA),
                (false, Vop::ZA, Vop::ZA),
                (false, Vop::IA, Vop::IB),
                (false, Vop::ZA, Vop::XB),
                (false, Vop::ZA, Vop::YB),
                (false, Vop::IA, Vop::YB),
                (true, Vop::XB, Vop::YF),
                (true, Vop::XB, Vop::ZF),
                (true, Vop::XB, Vop::IF),
                (true, Vop::XB, Vop::XF),
                (false, Vop::YB, Vop::YA),
                (false, Vop::XB, Vop::YA),
                (false, Vop::YB, Vop::IA),
                (false, Vop::XB, Vop::IA),
                (false, Vop::XB, Vop::YB),
                (false, Vop::YB, Vop::YB),
                (false, Vop::XB, Vop::IB),
                (false, Vop::YB, Vop::IB),
                (true, Vop::XB, Vop::ZC),
                (true, Vop::XB, Vop::YC),
                (true, Vop::XB, Vop::XC),
                (true, Vop::XB, Vop::IC),
            ],
            [
                (false, Vop::IB, Vop::ZA),
                (false, Vop::YB, Vop::YA),
                (false, Vop::YB, Vop::IA),
                (false, Vop::IB, Vop::IA),
                (false, Vop::YB, Vop::YB),
                (false, Vop::IB, Vop::YB),
                (false, Vop::IB, Vop::XB),
                (false, Vop::YB, Vop::IB),
                (true, Vop::IA, Vop::XF),
                (true, Vop::IA, Vop::IF),
                (true, Vop::IA, Vop::ZF),
                (true, Vop::IA, Vop::YF),
                (false, Vop::IA, Vop::IA),
                (false, Vop::YA, Vop::IA),
                (false, Vop::IA, Vop::YA),
                (false, Vop::YA, Vop::YA),
                (false, Vop::YA, Vop::IB),
                (false, Vop::IA, Vop::IB),
                (false, Vop::YA, Vop::YB),
                (false, Vop::IA, Vop::YB),
                (true, Vop::IA, Vop::IC),
                (true, Vop::IA, Vop::XC),
                (true, Vop::IA, Vop::YC),
                (true, Vop::IA, Vop::ZC),
            ],
            [
                (false, Vop::XB, Vop::IA),
                (false, Vop::YB, Vop::IA),
                (false, Vop::YB, Vop::YA),
                (false, Vop::XB, Vop::ZA),
                (false, Vop::YB, Vop::IB),
                (false, Vop::XB, Vop::XB),
                (false, Vop::XB, Vop::YB),
                (false, Vop::YB, Vop::YB),
                (true, Vop::IA, Vop::YF),
                (true, Vop::IA, Vop::ZF),
                (true, Vop::IA, Vop::IF),
                (true, Vop::IA, Vop::XF),
                (false, Vop::ZA, Vop::YA),
                (false, Vop::IA, Vop::YA),
                (false, Vop::ZA, Vop::IA),
                (false, Vop::IA, Vop::IA),
                (false, Vop::IA, Vop::YB),
                (false, Vop::ZA, Vop::YB),
                (false, Vop::IA, Vop::IB),
                (false, Vop::ZA, Vop::IB),
                (true, Vop::IA, Vop::ZC),
                (true, Vop::IA, Vop::YC),
                (true, Vop::IA, Vop::XC),
                (true, Vop::IA, Vop::IC),
            ],
            [
                (false, Vop::YB, Vop::IA),
                (false, Vop::XB, Vop::IA),
                (false, Vop::XB, Vop::YA),
                (false, Vop::YB, Vop::ZA),
                (false, Vop::XB, Vop::IB),
                (false, Vop::YB, Vop::XB),
                (false, Vop::YB, Vop::YB),
                (false, Vop::XB, Vop::YB),
                (true, Vop::IA, Vop::ZF),
                (true, Vop::IA, Vop::YF),
                (true, Vop::IA, Vop::XF),
                (true, Vop::IA, Vop::IF),
                (false, Vop::IA, Vop::YA),
                (false, Vop::ZA, Vop::YA),
                (false, Vop::IA, Vop::IA),
                (false, Vop::ZA, Vop::IA),
                (false, Vop::ZA, Vop::YB),
                (false, Vop::IA, Vop::YB),
                (false, Vop::ZA, Vop::IB),
                (false, Vop::IA, Vop::IB),
                (true, Vop::IA, Vop::YC),
                (true, Vop::IA, Vop::ZC),
                (true, Vop::IA, Vop::IC),
                (true, Vop::IA, Vop::XC),
            ],
            [
                (false, Vop::YB, Vop::ZA),
                (false, Vop::IB, Vop::YA),
                (false, Vop::IB, Vop::IA),
                (false, Vop::YB, Vop::IA),
                (false, Vop::IB, Vop::YB),
                (false, Vop::YB, Vop::YB),
                (false, Vop::YB, Vop::XB),
                (false, Vop::IB, Vop::IB),
                (true, Vop::IA, Vop::IF),
                (true, Vop::IA, Vop::XF),
                (true, Vop::IA, Vop::YF),
                (true, Vop::IA, Vop::ZF),
                (false, Vop::YA, Vop::IA),
                (false, Vop::IA, Vop::IA),
                (false, Vop::YA, Vop::YA),
                (false, Vop::IA, Vop::YA),
                (false, Vop::IA, Vop::IB),
                (false, Vop::YA, Vop::IB),
                (false, Vop::IA, Vop::YB),
                (false, Vop::YA, Vop::YB),
                (true, Vop::IA, Vop::XC),
                (true, Vop::IA, Vop::IC),
                (true, Vop::IA, Vop::ZC),
                (true, Vop::IA, Vop::YC),
            ],
            [
                (true, Vop::YF, Vop::YB),
                (true, Vop::IF, Vop::XB),
                (true, Vop::IF, Vop::YB),
                (true, Vop::YF, Vop::XB),
                (true, Vop::IF, Vop::ZA),
                (true, Vop::YF, Vop::IA),
                (true, Vop::YF, Vop::ZA),
                (true, Vop::IF, Vop::IA),
                (false, Vop::IA, Vop::IA),
                (false, Vop::IA, Vop::YA),
                (false, Vop::YA, Vop::YA),
                (false, Vop::YA, Vop::IA),
                (false, Vop::YB, Vop::YB),
                (false, Vop::IB, Vop::IB),
                (false, Vop::YB, Vop::IB),
                (false, Vop::IB, Vop::YB),
                (false, Vop::IB, Vop::YA),
                (false, Vop::YB, Vop::IA),
                (false, Vop::IB, Vop::IA),
                (false, Vop::YB, Vop::YA),
                (false, Vop::YA, Vop::IB),
                (false, Vop::YA, Vop::YB),
                (false, Vop::IA, Vop::YB),
                (false, Vop::IA, Vop::IB),
            ],
            [
                (true, Vop::YF, Vop::XB),
                (true, Vop::IF, Vop::YB),
                (true, Vop::IF, Vop::XB),
                (true, Vop::YF, Vop::YB),
                (true, Vop::IF, Vop::IA),
                (true, Vop::YF, Vop::ZA),
                (true, Vop::YF, Vop::IA),
                (true, Vop::IF, Vop::ZA),
                (false, Vop::YA, Vop::IA),
                (false, Vop::YA, Vop::YA),
                (false, Vop::IA, Vop::YA),
                (false, Vop::IA, Vop::IA),
                (false, Vop::IB, Vop::YB),
                (false, Vop::YB, Vop::IB),
                (false, Vop::IB, Vop::IB),
                (false, Vop::YB, Vop::YB),
                (false, Vop::YB, Vop::YA),
                (false, Vop::IB, Vop::IA),
                (false, Vop::YB, Vop::IA),
                (false, Vop::IB, Vop::YA),
                (false, Vop::IA, Vop::IB),
                (false, Vop::IA, Vop::YB),
                (false, Vop::YA, Vop::YB),
                (false, Vop::YA, Vop::IB),
            ],
            [
                (true, Vop::IF, Vop::YB),
                (true, Vop::IF, Vop::ZB),
                (true, Vop::IF, Vop::IB),
                (true, Vop::IF, Vop::XB),
                (true, Vop::IF, Vop::XA),
                (true, Vop::IF, Vop::IA),
                (true, Vop::IF, Vop::ZA),
                (true, Vop::IF, Vop::YA),
                (false, Vop::YA, Vop::YA),
                (false, Vop::YA, Vop::IA),
                (false, Vop::IA, Vop::IA),
                (false, Vop::IA, Vop::YA),
                (false, Vop::YB, Vop::IB),
                (false, Vop::IB, Vop::YB),
                (false, Vop::YB, Vop::YB),
                (false, Vop::IB, Vop::IB),
                (false, Vop::IB, Vop::IA),
                (false, Vop::YB, Vop::YA),
                (false, Vop::IB, Vop::YA),
                (false, Vop::YB, Vop::IA),
                (false, Vop::IA, Vop::YB),
                (false, Vop::IA, Vop::IB),
                (false, Vop::YA, Vop::IB),
                (false, Vop::YA, Vop::YB),
            ],
            [
                (true, Vop::IF, Vop::XB),
                (true, Vop::IF, Vop::IB),
                (true, Vop::IF, Vop::ZB),
                (true, Vop::IF, Vop::YB),
                (true, Vop::IF, Vop::YA),
                (true, Vop::IF, Vop::ZA),
                (true, Vop::IF, Vop::IA),
                (true, Vop::IF, Vop::XA),
                (false, Vop::IA, Vop::YA),
                (false, Vop::IA, Vop::IA),
                (false, Vop::YA, Vop::IA),
                (false, Vop::YA, Vop::YA),
                (false, Vop::IB, Vop::IB),
                (false, Vop::YB, Vop::YB),
                (false, Vop::IB, Vop::YB),
                (false, Vop::YB, Vop::IB),
                (false, Vop::YB, Vop::IA),
                (false, Vop::IB, Vop::YA),
                (false, Vop::YB, Vop::YA),
                (false, Vop::IB, Vop::IA),
                (false, Vop::YA, Vop::YB),
                (false, Vop::YA, Vop::IB),
                (false, Vop::IA, Vop::IB),
                (false, Vop::IA, Vop::YB),
            ],
            [
                (false, Vop::YA, Vop::XB),
                (false, Vop::IA, Vop::YB),
                (false, Vop::IA, Vop::IB),
                (false, Vop::YA, Vop::YB),
                (false, Vop::IA, Vop::IA),
                (false, Vop::YA, Vop::ZA),
                (false, Vop::YA, Vop::IA),
                (false, Vop::IA, Vop::YA),
                (false, Vop::YB, Vop::YB),
                (false, Vop::YB, Vop::IB),
                (false, Vop::IB, Vop::YB),
                (false, Vop::IB, Vop::IB),
                (true, Vop::IE, Vop::YE),
                (true, Vop::IE, Vop::ZE),
                (true, Vop::IE, Vop::IE),
                (true, Vop::IE, Vop::XE),
                (true, Vop::IE, Vop::ID),
                (true, Vop::IE, Vop::XD),
                (true, Vop::IE, Vop::YD),
                (true, Vop::IE, Vop::ZD),
                (false, Vop::IB, Vop::YA),
                (false, Vop::IB, Vop::IA),
                (false, Vop::YB, Vop::YA),
                (false, Vop::YB, Vop::IA),
            ],
            [
                (false, Vop::YA, Vop::YB),
                (false, Vop::IA, Vop::IB),
                (false, Vop::IA, Vop::YB),
                (false, Vop::YA, Vop::XB),
                (false, Vop::IA, Vop::YA),
                (false, Vop::YA, Vop::IA),
                (false, Vop::YA, Vop::ZA),
                (false, Vop::IA, Vop::IA),
                (false, Vop::IB, Vop::IB),
                (false, Vop::IB, Vop::YB),
                (false, Vop::YB, Vop::IB),
                (false, Vop::YB, Vop::YB),
                (true, Vop::IE, Vop::XE),
                (true, Vop::IE, Vop::IE),
                (true, Vop::IE, Vop::ZE),
                (true, Vop::IE, Vop::YE),
                (true, Vop::IE, Vop::ZD),
                (true, Vop::IE, Vop::YD),
                (true, Vop::IE, Vop::XD),
                (true, Vop::IE, Vop::ID),
                (false, Vop::YB, Vop::IA),
                (false, Vop::YB, Vop::YA),
                (false, Vop::IB, Vop::IA),
                (false, Vop::IB, Vop::YA),
            ],
            [
                (false, Vop::IA, Vop::XB),
                (false, Vop::YA, Vop::YB),
                (false, Vop::YA, Vop::IB),
                (false, Vop::IA, Vop::YB),
                (false, Vop::YA, Vop::IA),
                (false, Vop::IA, Vop::ZA),
                (false, Vop::IA, Vop::IA),
                (false, Vop::YA, Vop::YA),
                (false, Vop::IB, Vop::YB),
                (false, Vop::IB, Vop::IB),
                (false, Vop::YB, Vop::YB),
                (false, Vop::YB, Vop::IB),
                (true, Vop::IE, Vop::IE),
                (true, Vop::IE, Vop::XE),
                (true, Vop::IE, Vop::YE),
                (true, Vop::IE, Vop::ZE),
                (true, Vop::IE, Vop::YD),
                (true, Vop::IE, Vop::ZD),
                (true, Vop::IE, Vop::ID),
                (true, Vop::IE, Vop::XD),
                (false, Vop::YB, Vop::YA),
                (false, Vop::YB, Vop::IA),
                (false, Vop::IB, Vop::YA),
                (false, Vop::IB, Vop::IA),
            ],
            [
                (false, Vop::IA, Vop::YB),
                (false, Vop::YA, Vop::IB),
                (false, Vop::YA, Vop::YB),
                (false, Vop::IA, Vop::XB),
                (false, Vop::YA, Vop::YA),
                (false, Vop::IA, Vop::IA),
                (false, Vop::IA, Vop::ZA),
                (false, Vop::YA, Vop::IA),
                (false, Vop::YB, Vop::IB),
                (false, Vop::YB, Vop::YB),
                (false, Vop::IB, Vop::IB),
                (false, Vop::IB, Vop::YB),
                (true, Vop::IE, Vop::ZE),
                (true, Vop::IE, Vop::YE),
                (true, Vop::IE, Vop::XE),
                (true, Vop::IE, Vop::IE),
                (true, Vop::IE, Vop::XD),
                (true, Vop::IE, Vop::ID),
                (true, Vop::IE, Vop::ZD),
                (true, Vop::IE, Vop::YD),
                (false, Vop::IB, Vop::IA),
                (false, Vop::IB, Vop::YA),
                (false, Vop::YB, Vop::IA),
                (false, Vop::YB, Vop::YA),
            ],
            [
                (false, Vop::YB, Vop::YB),
                (false, Vop::IB, Vop::IB),
                (false, Vop::IB, Vop::YB),
                (false, Vop::YB, Vop::XB),
                (false, Vop::IB, Vop::YA),
                (false, Vop::YB, Vop::IA),
                (false, Vop::YB, Vop::ZA),
                (false, Vop::IB, Vop::IA),
                (false, Vop::YA, Vop::IB),
                (false, Vop::YA, Vop::YB),
                (false, Vop::IA, Vop::IB),
                (false, Vop::IA, Vop::YB),
                (true, Vop::ID, Vop::IE),
                (true, Vop::ID, Vop::XE),
                (true, Vop::ID, Vop::YE),
                (true, Vop::ID, Vop::ZE),
                (true, Vop::ID, Vop::YD),
                (true, Vop::ID, Vop::ZD),
                (true, Vop::ID, Vop::ID),
                (true, Vop::ID, Vop::XD),
                (false, Vop::IA, Vop::IA),
                (false, Vop::IA, Vop::YA),
                (false, Vop::YA, Vop::IA),
                (false, Vop::YA, Vop::YA),
            ],
            [
                (false, Vop::YB, Vop::XB),
                (false, Vop::IB, Vop::YB),
                (false, Vop::IB, Vop::IB),
                (false, Vop::YB, Vop::YB),
                (false, Vop::IB, Vop::IA),
                (false, Vop::YB, Vop::ZA),
                (false, Vop::YB, Vop::IA),
                (false, Vop::IB, Vop::YA),
                (false, Vop::IA, Vop::YB),
                (false, Vop::IA, Vop::IB),
                (false, Vop::YA, Vop::YB),
                (false, Vop::YA, Vop::IB),
                (true, Vop::ID, Vop::ZE),
                (true, Vop::ID, Vop::YE),
                (true, Vop::ID, Vop::XE),
                (true, Vop::ID, Vop::IE),
                (true, Vop::ID, Vop::XD),
                (true, Vop::ID, Vop::ID),
                (true, Vop::ID, Vop::ZD),
                (true, Vop::ID, Vop::YD),
                (false, Vop::YA, Vop::YA),
                (false, Vop::YA, Vop::IA),
                (false, Vop::IA, Vop::YA),
                (false, Vop::IA, Vop::IA),
            ],
            [
                (false, Vop::IB, Vop::YB),
                (false, Vop::YB, Vop::IB),
                (false, Vop::YB, Vop::YB),
                (false, Vop::IB, Vop::XB),
                (false, Vop::YB, Vop::YA),
                (false, Vop::IB, Vop::IA),
                (false, Vop::IB, Vop::ZA),
                (false, Vop::YB, Vop::IA),
                (false, Vop::IA, Vop::IB),
                (false, Vop::IA, Vop::YB),
                (false, Vop::YA, Vop::IB),
                (false, Vop::YA, Vop::YB),
                (true, Vop::ID, Vop::YE),
                (true, Vop::ID, Vop::ZE),
                (true, Vop::ID, Vop::IE),
                (true, Vop::ID, Vop::XE),
                (true, Vop::ID, Vop::ID),
                (true, Vop::ID, Vop::XD),
                (true, Vop::ID, Vop::YD),
                (true, Vop::ID, Vop::ZD),
                (false, Vop::YA, Vop::IA),
                (false, Vop::YA, Vop::YA),
                (false, Vop::IA, Vop::IA),
                (false, Vop::IA, Vop::YA),
            ],
            [
                (false, Vop::IB, Vop::XB),
                (false, Vop::YB, Vop::YB),
                (false, Vop::YB, Vop::IB),
                (false, Vop::IB, Vop::YB),
                (false, Vop::YB, Vop::IA),
                (false, Vop::IB, Vop::ZA),
                (false, Vop::IB, Vop::IA),
                (false, Vop::YB, Vop::YA),
                (false, Vop::YA, Vop::YB),
                (false, Vop::YA, Vop::IB),
                (false, Vop::IA, Vop::YB),
                (false, Vop::IA, Vop::IB),
                (true, Vop::ID, Vop::XE),
                (true, Vop::ID, Vop::IE),
                (true, Vop::ID, Vop::ZE),
                (true, Vop::ID, Vop::YE),
                (true, Vop::ID, Vop::ZD),
                (true, Vop::ID, Vop::YD),
                (true, Vop::ID, Vop::XD),
                (true, Vop::ID, Vop::ID),
                (false, Vop::IA, Vop::YA),
                (false, Vop::IA, Vop::IA),
                (false, Vop::YA, Vop::YA),
                (false, Vop::YA, Vop::IA),
            ],
            [
                (true, Vop::YC, Vop::XB),
                (true, Vop::IC, Vop::YB),
                (true, Vop::IC, Vop::XB),
                (true, Vop::YC, Vop::YB),
                (true, Vop::IC, Vop::IA),
                (true, Vop::YC, Vop::ZA),
                (true, Vop::YC, Vop::IA),
                (true, Vop::IC, Vop::ZA),
                (false, Vop::IB, Vop::YA),
                (false, Vop::IB, Vop::IA),
                (false, Vop::YB, Vop::IA),
                (false, Vop::YB, Vop::YA),
                (false, Vop::YA, Vop::IB),
                (false, Vop::IA, Vop::YB),
                (false, Vop::YA, Vop::YB),
                (false, Vop::IA, Vop::IB),
                (false, Vop::IA, Vop::IA),
                (false, Vop::YA, Vop::YA),
                (false, Vop::IA, Vop::YA),
                (false, Vop::YA, Vop::IA),
                (false, Vop::YB, Vop::YB),
                (false, Vop::YB, Vop::IB),
                (false, Vop::IB, Vop::IB),
                (false, Vop::IB, Vop::YB),
            ],
            [
                (true, Vop::YC, Vop::YB),
                (true, Vop::IC, Vop::XB),
                (true, Vop::IC, Vop::YB),
                (true, Vop::YC, Vop::XB),
                (true, Vop::IC, Vop::ZA),
                (true, Vop::YC, Vop::IA),
                (true, Vop::YC, Vop::ZA),
                (true, Vop::IC, Vop::IA),
                (false, Vop::YB, Vop::YA),
                (false, Vop::YB, Vop::IA),
                (false, Vop::IB, Vop::IA),
                (false, Vop::IB, Vop::YA),
                (false, Vop::IA, Vop::IB),
                (false, Vop::YA, Vop::YB),
                (false, Vop::IA, Vop::YB),
                (false, Vop::YA, Vop::IB),
                (false, Vop::YA, Vop::IA),
                (false, Vop::IA, Vop::YA),
                (false, Vop::YA, Vop::YA),
                (false, Vop::IA, Vop::IA),
                (false, Vop::IB, Vop::YB),
                (false, Vop::IB, Vop::IB),
                (false, Vop::YB, Vop::IB),
                (false, Vop::YB, Vop::YB),
            ],
            [
                (true, Vop::IC, Vop::XB),
                (true, Vop::IC, Vop::IB),
                (true, Vop::IC, Vop::ZB),
                (true, Vop::IC, Vop::YB),
                (true, Vop::IC, Vop::YA),
                (true, Vop::IC, Vop::ZA),
                (true, Vop::IC, Vop::IA),
                (true, Vop::IC, Vop::XA),
                (false, Vop::YB, Vop::IA),
                (false, Vop::YB, Vop::YA),
                (false, Vop::IB, Vop::YA),
                (false, Vop::IB, Vop::IA),
                (false, Vop::YA, Vop::YB),
                (false, Vop::IA, Vop::IB),
                (false, Vop::YA, Vop::IB),
                (false, Vop::IA, Vop::YB),
                (false, Vop::IA, Vop::YA),
                (false, Vop::YA, Vop::IA),
                (false, Vop::IA, Vop::IA),
                (false, Vop::YA, Vop::YA),
                (false, Vop::IB, Vop::IB),
                (false, Vop::IB, Vop::YB),
                (false, Vop::YB, Vop::YB),
                (false, Vop::YB, Vop::IB),
            ],
            [
                (true, Vop::IC, Vop::YB),
                (true, Vop::IC, Vop::ZB),
                (true, Vop::IC, Vop::IB),
                (true, Vop::IC, Vop::XB),
                (true, Vop::IC, Vop::XA),
                (true, Vop::IC, Vop::IA),
                (true, Vop::IC, Vop::ZA),
                (true, Vop::IC, Vop::YA),
                (false, Vop::IB, Vop::IA),
                (false, Vop::IB, Vop::YA),
                (false, Vop::YB, Vop::YA),
                (false, Vop::YB, Vop::IA),
                (false, Vop::IA, Vop::YB),
                (false, Vop::YA, Vop::IB),
                (false, Vop::IA, Vop::IB),
                (false, Vop::YA, Vop::YB),
                (false, Vop::YA, Vop::YA),
                (false, Vop::IA, Vop::IA),
                (false, Vop::YA, Vop::IA),
                (false, Vop::IA, Vop::YA),
                (false, Vop::YB, Vop::IB),
                (false, Vop::YB, Vop::YB),
                (false, Vop::IB, Vop::YB),
                (false, Vop::IB, Vop::IB),
            ],
        ],
    ];

    const X_GATE: Vop = Vop::XA;
    const Y_GATE: Vop = Vop::YA;
    const Z_GATE: Vop = Vop::ZA;
    const H_GATE: Vop = Vop::YC;
    const S_GATE: Vop = Vop::YB;
    const SDAG_GATE: Vop = Vop::XB;

    const SYMMETRIES: usize = 24;
    const MEAS_AXES: usize = 3;

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
        adjacent: Vec<NodeIdx>,
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
                adjacent: Vec::new(),
                vop: Vop::YC,
            }
        }
    }

    impl Node {
        fn len(&self) -> usize {
            self.adjacent.len()
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
            let other: NodeIdx = self[node].adjacent[0];

            match res {
                MeasurementResult::PlusOne => {
                    self[other].vop = self[other].vop * Vop::ZC;
                    let size = self[node].len();
                    for i in 0..size {
                        let third = self[node].adjacent[i];
                        if third != other && !self[other].adjacent.contains(&third) {
                            self[third].vop = self[third].vop * Z_GATE;
                        }
                    }
                }
                MeasurementResult::MinusOne => {
                    self[other].vop = self[other].vop * Vop::XC;
                    self[node].vop = self[node].vop * Vop::ZA;

                    let size = self[other].len();
                    for i in 0..size {
                        let third = self[other].adjacent[i];
                        if third != node && !self[node].adjacent.contains(&third) {
                            self[third].vop = self[third].vop * Z_GATE;
                        }
                    }
                }
            }

            let node_nbs = self[node].adjacent.clone();
            let other_nbs = self[other].adjacent.clone();

            let mut procced_edges: HashSet<(NodeIdx, NodeIdx)> = HashSet::new();
            let nlen = node_nbs.len();
            let olen = other_nbs.len();
            for i in 0..nlen {
                let nval = node_nbs[i];
                for j in 0..olen {
                    let oval = other_nbs[j];
                    let combined = match nval < oval {
                        true => (nval, oval),
                        false => (oval, nval),
                    };
                    if nval != oval && !procced_edges.contains(&combined) {
                        procced_edges.insert(combined);
                        self.toggle_edge(combined.0, combined.1);
                    }
                }
            }

            let mut intersection = Vec::new();
            for i in 0..nlen {
                if other_nbs.contains(&node_nbs[i]) {
                    intersection.push(node_nbs[i]);
                }
            }

            let ilen = intersection.len();
            for i in 0..ilen {
                for j in i + 1..ilen {
                    self.toggle_edge(intersection[i], intersection[j]);
                }
            }

            for i in 0..nlen {
                let nadj_i = node_nbs[i];
                if nadj_i != other {
                    self.toggle_edge(other, nadj_i);
                }
            }

            (res, false)
        }
        fn int_measure_y(&mut self, node: NodeIdx) -> MeasurementResult {
            let res = rand::rng().random();

            let nlen = self[node].len();
            for i in 0..nlen {
                let other = self[node].adjacent[i];
                match res {
                    MeasurementResult::PlusOne => self[other].vop = self[other].vop * S_GATE,
                    MeasurementResult::MinusOne => self[other].vop = self[other].vop * SDAG_GATE,
                }
            }

            for i in 0..nlen {
                let nval = self[node].adjacent[i];
                for j in i + 1..=nlen {
                    let oval = if j == nlen {
                        node
                    } else {
                        self[node].adjacent[j]
                    };
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

            let nlen = self[node].len();
            for i in 0..nlen {
                let other = self[node].adjacent[i];
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
        fn local_comp(&mut self, node: NodeIdx) {
            let len = self[node].len();
            for i in 0..len {
                for j in i + 1..len {
                    self.toggle_edge(self[node].adjacent[i], self[node].adjacent[j]);
                }
                let inode = self[node].adjacent[i];
                self[inode].vop = self[inode].vop * S_GATE;
            }
            self[node].vop = self[node].vop * Vop::YD;
        }
        fn toggle_edge(&mut self, na: NodeIdx, nb: NodeIdx) -> bool {
            let lba = self[na].len();
            let lbb = self[nb].len();
            self[na].adjacent.retain(|&v| v != nb);
            self[nb].adjacent.retain(|&v| v != na);
            if lba == self[na].len() {
                debug_assert_eq!(lbb, self[nb].len());
                self[na].adjacent.push(nb);
                self[nb].adjacent.push(na);
                false
            } else {
                true
            }
        }
        fn delete_edge(&mut self, na: NodeIdx, nb: NodeIdx) {
            self[na].adjacent.retain(|&v| v != nb);
            self[nb].adjacent.retain(|&v| v != na);
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
        fn x(&mut self, qubit: NodeIdx) {
            self[qubit].vop = X_GATE * self[qubit].vop;
        }

        /// Apply a Y (Pauli-Y) gate to the given qubit.
        ///
        /// `node` is the index of the qubit.
        fn y(&mut self, qubit: NodeIdx) {
            self[qubit].vop = Y_GATE * self[qubit].vop;
        }

        /// Apply a Z (Pauli-Z) gate to the given qubit.
        ///
        /// `node` is the index of the qubit.
        fn z(&mut self, qubit: NodeIdx) {
            self[qubit].vop = Z_GATE * self[qubit].vop;
        }

        /// Apply an H (Hadamard) gate to the given qubit.
        ///
        /// `node` is the index of the qubit.
        fn h(&mut self, qubit: NodeIdx) {
            self[qubit].vop = H_GATE * self[qubit].vop;
        }

        /// Apply an S (phase) gate to the given qubit.
        ///
        /// `node` is the index of the qubit.
        fn s(&mut self, qubit: NodeIdx) {
            self[qubit].vop = S_GATE * self[qubit].vop;
        }

        /// Apply an S† (inverse phase) gate to the given qubit.
        ///
        /// `node` is the index of the qubit.
        fn sdag(&mut self, qubit: NodeIdx) {
            self[qubit].vop = SDAG_GATE * self[qubit].vop;
        }

        /// Apply a controlled-Z (CZ) gate with `control` and `target` qubits.
        fn cz(&mut self, control: NodeIdx, target: NodeIdx) {
            let c_has_t = self[control].len() > 1
                || (self[control].len() == 1 && self[control].adjacent[0] != target);
            let t_has_c = self[target].len() > 1
                || (self[target].len() == 1 && self[target].adjacent[0] != control);

            if c_has_t {
                self.remove_vop(control, target);
            }
            if t_has_c {
                self.remove_vop(target, control);
            }
            if c_has_t && !self[control].vop.is_in_z() {
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
                self[control].adjacent.push(target);
                self[target].adjacent.push(control);
            } else {
                self[control].adjacent.retain(|&v| v != target);
                self[target].adjacent.retain(|&v| v != control);
            }
            self[control].vop = val.1;
            self[target].vop = val.2;
        }

        /// Apply a controlled-X (CX) / CNOT gate with `control` and `target`.
        fn cx(&mut self, control: NodeIdx, target: NodeIdx) {
            self.h(target);
            self.cz(control, target);
            self.h(target);
        }

        /// Apply an X-controlled X gate (CX in the X basis).
        fn xcx(&mut self, control: NodeIdx, target: NodeIdx) {
            self.h(control);
            self.cx(control, target);
            self.h(control);
        }

        /// Apply a Y-controlled X gate (control qubit in the Y basis).
        fn ycx(&mut self, control: NodeIdx, target: NodeIdx) {
            self.sdag(control);
            self.xcx(control, target);
            self.s(control);
        }

        /// Apply an X-controlled Z gate (target in X basis).
        fn xcz(&mut self, control: NodeIdx, target: NodeIdx) {
            self.cx(target, control);
        }

        /// Apply a Y-controlled Z gate (target in Y basis).
        fn ycz(&mut self, control: NodeIdx, target: NodeIdx) {
            self.cy(target, control);
        }

        /// Apply a controlled-Y (CY) gate with `control` and `target`.
        fn cy(&mut self, control: NodeIdx, target: NodeIdx) {
            self.sdag(target);
            self.cx(control, target);
            self.s(target);
        }

        /// Apply an X-controlled Y gate (control in X basis).
        fn xcy(&mut self, control: NodeIdx, target: NodeIdx) {
            self.ycx(target, control);
        }

        /// Apply a Y-controlled Y gate (both in Y basis).
        fn ycy(&mut self, control: NodeIdx, target: NodeIdx) {
            self.sdag(target);
            self.ycx(control, target);
            self.s(target);
        }

        /// Perform a projective measurement of `qubit` in the X basis.
        ///
        /// Returns `MeasurementResult.PlusOne` or `MeasurementResult.MinusOne`.
        fn measure_x(&mut self, qubit: NodeIdx) -> MeasurementResult {
            let (res, _) = self.measure(qubit, Axis::X);
            res
        }

        /// Perform a projective measurement of `qubit` in the Y basis.
        ///
        /// Returns `MeasurementResult.PlusOne` or `MeasurementResult.MinusOne`.
        fn measure_y(&mut self, qubit: NodeIdx) -> MeasurementResult {
            let (res, _) = self.measure(qubit, Axis::Y);
            res
        }

        /// Perform a projective measurement of `qubit` in the Z basis.
        ///
        /// Returns `MeasurementResult.PlusOne` or `MeasurementResult.MinusOne`.
        fn measure_z(&mut self, qubit: NodeIdx) -> MeasurementResult {
            let (res, _) = self.measure(qubit, Axis::Z);
            res
        }

        /// Return the set of qubits that are entangled with `qubit`.
        ///
        /// This follows adjacency in the underlying graph.
        fn get_entangled_group(&self, qubit: NodeIdx) -> HashSet<NodeIdx> {
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
        fn peek_measure_set(
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
