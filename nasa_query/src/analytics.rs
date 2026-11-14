use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct StatisticalSummary {
    pub count: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub slope: f64,
    pub intercept: f64,
}

/// Computes statistical metrics over paired (x, y) observations:
/// - Mean: μ = Σ y_i / N
/// - Standard Deviation: Sample standard deviation (Bessel's correction N - 1)
/// - Linear Regression: OLS slope β = Cov(X, Y) / Var(X) and intercept α = μ_y - β * μ_x
pub fn compute_statistics(x: &[f64], y: &[f64]) -> Option<StatisticalSummary> {
    let n = y.len();
    if n == 0 || x.len() != n {
        return None;
    }

    let n_f64 = n as f64;
    let mean_y = y.iter().copied().sum::<f64>() / n_f64;
    let mean_x = x.iter().copied().sum::<f64>() / n_f64;

    // Sample variance (Bessel's correction when n > 1, 0.0 when n <= 1)
    let variance = if n > 1 {
        let sum_sq_diff: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        sum_sq_diff / ((n - 1) as f64)
    } else {
        0.0
    };
    let std_dev = variance.sqrt();

    // Ordinary Least Squares (OLS) Linear Regression
    let mut cov_xy = 0.0;
    let mut var_x = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        cov_xy += (xi - mean_x) * (yi - mean_y);
        var_x += (xi - mean_x).powi(2);
    }

    let (slope, intercept) = if var_x > f64::EPSILON {
        let b = cov_xy / var_x;
        let a = mean_y - b * mean_x;
        (b, a)
    } else {
        (0.0, mean_y)
    };

    Some(StatisticalSummary {
        count: n,
        mean: mean_y,
        std_dev,
        slope,
        intercept,
    })
}

/// Computes Z-score = (value - mean) / std_dev.
/// Safely returns 0.0 if standard deviation is below numerical epsilon.
#[inline]
pub fn compute_z_score(val: f64, mean: f64, std_dev: f64) -> f64 {
    if std_dev > f64::EPSILON {
        (val - mean) / std_dev
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_or_mismatched() {
        assert_eq!(compute_statistics(&[], &[]), None);
        assert_eq!(compute_statistics(&[1.0], &[1.0, 2.0]), None);
    }

    #[test]
    fn test_linear_trend_and_stats() {
        // y = 2x + 1
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];

        let stats = compute_statistics(&x, &y).unwrap();
        assert_eq!(stats.count, 5);
        assert!((stats.mean - 7.0).abs() < 1e-9);
        assert!((stats.slope - 2.0).abs() < 1e-9);
        assert!((stats.intercept - 1.0).abs() < 1e-9);

        // Sample variance of [3, 5, 7, 9, 11] is (16 + 4 + 0 + 4 + 16) / 4 = 10 -> std_dev = sqrt(10) ≈ 3.162277
        assert!((stats.std_dev - 10.0_f64.sqrt()).abs() < 1e-6);

        // Z-score of middle point should be 0.0
        assert!((compute_z_score(7.0, stats.mean, stats.std_dev)).abs() < 1e-9);
        // Z-score of 11.0: (11 - 7) / sqrt(10) ≈ 1.2649
        assert!((compute_z_score(11.0, stats.mean, stats.std_dev) - 4.0 / 10.0_f64.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_zero_variance() {
        let x = vec![2000.0, 2001.0, 2002.0];
        let y = vec![5.0, 5.0, 5.0];

        let stats = compute_statistics(&x, &y).unwrap();
        assert_eq!(stats.std_dev, 0.0);
        assert_eq!(stats.slope, 0.0);
        assert_eq!(stats.intercept, 5.0);
        assert_eq!(compute_z_score(5.0, stats.mean, stats.std_dev), 0.0);
    }
}
