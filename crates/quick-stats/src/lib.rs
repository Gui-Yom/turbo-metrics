pub mod full {
    #[derive(Debug)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize))]
    pub struct Stats {
        pub min: f64,
        pub max: f64,
        pub mean: f64,
        /// Population variance
        pub var: f64,
        /// Sample variance
        pub sample_var: f64,
        pub stddev: f64,
        pub sample_stddev: f64,
        pub p1: f64,
        pub p5: f64,
        pub p50: f64,
        pub p95: f64,
        pub p99: f64,
    }

    impl Stats {
        pub fn compute(values: &[f64]) -> Self {
            let mut sorted = values.to_vec();
            sorted.sort_unstable_by(|a, b| a.total_cmp(b));
            let min = sorted[0];
            let max = sorted[sorted.len() - 1];
            let mean = sorted.iter().sum::<f64>() / values.len() as f64;
            let var = compute_var(values, mean, false);
            let sample_var = compute_var(values, mean, true);
            let stddev = var.sqrt();
            let sample_stddev = sample_var.sqrt();
            let p1 = percentile_of_sorted(&sorted, 1.0);
            let p5 = percentile_of_sorted(&sorted, 5.0);
            let p50 = percentile_of_sorted(&sorted, 50.0);
            let p95 = percentile_of_sorted(&sorted, 95.0);
            let p99 = percentile_of_sorted(&sorted, 99.0);
            Self {
                min,
                max,
                mean,
                var,
                sample_var,
                stddev,
                sample_stddev,
                p1,
                p5,
                p50,
                p95,
                p99,
            }
        }
    }

    // Helper function: extract a value representing the `pct` percentile of a sorted sample-set, using
    // linear interpolation. If samples are not sorted, return nonsensical value.
    fn percentile_of_sorted(sorted_samples: &[f64], pct: f64) -> f64 {
        assert!(!sorted_samples.is_empty());
        if sorted_samples.len() == 1 {
            return sorted_samples[0];
        }
        let zero: f64 = 0.0;
        assert!(zero <= pct);
        let hundred = 100_f64;
        assert!(pct <= hundred);
        if pct == hundred {
            return sorted_samples[sorted_samples.len() - 1];
        }
        let length = (sorted_samples.len() - 1) as f64;
        let rank = (pct / hundred) * length;
        let lrank = rank.floor();
        let d = rank - lrank;
        let n = lrank as usize;
        let lo = sorted_samples[n];
        let hi = sorted_samples[n + 1];
        lo + (hi - lo) * d
    }

    fn compute_var(values: &[f64], mean: f64, sample: bool) -> f64 {
        if values.len() < 2 {
            0.0
        } else {
            let mut v: f64 = 0.0;
            for s in values {
                let x = *s - mean;
                v += x * x;
            }
            // N.B., this is _supposed to be_ len-1, not len. If you
            // change it back to len, you will be calculating a
            // population variance, not a sample variance.
            let denom = if sample {
                values.len() - 1
            } else {
                values.len()
            } as f64;
            v / denom
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn it_works() {
            let values = [0.0, 1.0, 3.0, 4.0];
            dbg!(Stats::compute(&values));
        }
    }
}
