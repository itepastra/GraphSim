use std::{
    collections::HashMap,
    iter::zip,
    time::{Duration, Instant},
};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use graphsim::graphsim::GraphSim;

const BASE: usize = 10;
const SIZES: [usize; 11] = [
    BASE,
    2 * BASE,
    4 * BASE,
    8 * BASE,
    16 * BASE,
    32 * BASE,
    64 * BASE,
    128 * BASE,
    256 * BASE,
    512 * BASE,
    1024 * BASE,
];

fn create_qubits(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_qubits");
    for size in SIZES.iter() {
        group.throughput(criterion::Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| GraphSim::new(size));
        });
    }
    group.finish();
}

fn scatter_single_qubit_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("scatter_single_qubit_gates"));
    for size in SIZES.iter() {
        group.throughput(criterion::Throughput::Elements(*size as u64));
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter_custom(|iters| {
                //prepare
                let mut gs = GraphSim::new(*size);
                let qubits: Vec<usize> = (0..iters).map(|_| rand::random_range(0..*size)).collect();
                let start = Instant::now();
                for qb in qubits {
                    gs.h(qb);
                }
                start.elapsed()
            });
        });
    }
    group.finish();
}

fn scatter_two_qubit_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("scatter_two_qubit_gates"));
    for size in SIZES.iter() {
        group.throughput(criterion::Throughput::Elements(*size as u64));
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter_custom(|iters| {
                let mut gs = GraphSim::new(*size);
                let controls: Vec<usize> =
                    (0..iters).map(|_| rand::random_range(0..*size)).collect();
                let targets: Vec<usize> =
                    (0..iters).map(|_| rand::random_range(0..*size)).collect();
                let comb: Vec<(usize, usize)> = zip(controls, targets)
                    .map(|(c, t)| {
                        if c != t {
                            (c, t)
                        } else if t == 0 {
                            (c, t + 1)
                        } else {
                            (c, t - 1)
                        }
                    })
                    .collect();
                let pre_shuffle: Vec<usize> =
                    (0..*size).map(|_| rand::random_range(0..*size)).collect();
                for qubit in pre_shuffle {
                    match rand::random_range(0..5) {
                        0 => gs.h(qubit),
                        1 => gs.x(qubit),
                        2 => gs.y(qubit),
                        3 => gs.z(qubit),
                        4 => gs.s(qubit),
                        _ => {}
                    }
                }
                let start = Instant::now();
                for (c, t) in comb {
                    gs.cz(c, t);
                }
                start.elapsed()
            });
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(25)).warm_up_time(Duration::from_secs(3)).sample_size(250);
    targets = create_qubits, scatter_single_qubit_gates, scatter_two_qubit_gates
}
criterion_main!(benches);
