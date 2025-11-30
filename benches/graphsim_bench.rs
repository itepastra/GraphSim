use std::{collections::HashMap, iter::zip, time::Instant};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use graphsim::graphsim::GraphSim;

fn create_qubits(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_qubits");
    const BASE: usize = 10;
    for size in [
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
    ]
    .iter()
    {
        group.throughput(criterion::Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| GraphSim::new(size));
        });
    }
    group.finish();
}

fn scatter_single_qubit_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("scatter_single_qubit_gates"));
    const BASE: usize = 10;
    for size in [
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
    ]
    .iter()
    {
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
    const BASE: usize = 10;
    for size in [
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
    ]
    .iter()
    {
        let mut gs = GraphSim::new(*size);
        group.throughput(criterion::Throughput::Elements(*size as u64));
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter_custom(|iters| {
                //prepare
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
                let mut adjacents: HashMap<usize, usize> = HashMap::new();
                let start = Instant::now();
                for (c, t) in comb {
                    gs.cz(c, t);
                }
                let fin = start.elapsed();
                let mut hval: Vec<_> = adjacents.iter().collect();
                hval.sort();
                println!("histogram of {} iters: {:?}", iters, hval);
                fin
            });
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = create_qubits, scatter_single_qubit_gates, scatter_two_qubit_gates
}
criterion_main!(benches);
