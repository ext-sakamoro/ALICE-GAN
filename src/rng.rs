//! Simple PRNG (xorshift64) — no external deps.

// Simple PRNG (xorshift64) — no external deps
// ---------------------------------------------------------------------------

/// A simple xorshift64 pseudo-random number generator.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Create a new RNG with the given seed.
    ///
    /// # Panics
    ///
    /// Panics if `seed` is zero.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        assert!(seed != 0, "seed must be non-zero");
        Self { state: seed }
    }

    /// Returns the next `u64` value.
    pub const fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a `f64` in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Returns a `f64` sampled from an approximate standard normal distribution
    /// using the Box-Muller transform.
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Fill a slice with values drawn from N(0, 1).
    pub fn fill_normal(&mut self, buf: &mut [f64]) {
        for v in buf.iter_mut() {
            *v = self.next_normal();
        }
    }

    /// Fill a slice with values in `[0, 1)`.
    pub fn fill_uniform(&mut self, buf: &mut [f64]) {
        for v in buf.iter_mut() {
            *v = self.next_f64();
        }
    }
}
