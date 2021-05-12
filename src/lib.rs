pub use ndarray::{self, Array2};

pub trait Func: Fn(f64) -> f64 + Send + Sync {}
impl<T: Fn(f64) -> f64 + Send + Sync> Func for T {}

pub struct ActivationFunction<F: Func, DF: Func> {
    pub function: F,
    pub derivative: DF,
}

pub struct NeuralNetwork<F: Func, DF: Func> {
    activations: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,
    activation: ActivationFunction<F, DF>,
}

impl<F: Func, DF: Func> NeuralNetwork<F, DF> {
    pub fn new(layer_sizes: &[usize], activation: ActivationFunction<F, DF>) -> Self {
        let mut weights = Vec::with_capacity(layer_sizes.len());
        for (l, ln) in layer_sizes.iter().zip(layer_sizes.iter().skip(1)) {
            weights.push(Array2::from_shape_simple_fn((*ln, *l), rand::random));
        }

        Self {
            activations: layer_sizes
                .iter()
                .map(|size| Array2::zeros((*size, 1)))
                .collect(),
            weights,
            biases: layer_sizes
                .iter()
                .skip(1)
                .map(|size| Array2::from_shape_simple_fn((*size, 1), rand::random))
                .collect(),
            activation,
        }
    }

    pub fn predict(&self, input: &[f64]) -> Vec<f64> {
        let mut layer = Array2::from_shape_vec((input.len(), 1), input.to_vec()).unwrap();

        for (weight, bias) in self.weights.iter().zip(self.biases.iter()) {
            layer = weight.dot(&layer) + bias;
            layer.par_mapv_inplace(&self.activation.function);
        }

        layer.into_raw_vec()
    }

    fn feedforward(&mut self, input: &[f64]) {
        let mut layer = Array2::from_shape_vec((input.len(), 1), input.to_vec()).unwrap();
        self.activations[0].assign(&layer);

        for ((weight, bias), activation) in self
            .weights
            .iter()
            .zip(self.biases.iter())
            .zip(self.activations.iter_mut().skip(1))
        {
            layer = weight.dot(&layer) + bias;
            layer.par_mapv_inplace(&self.activation.function);
            activation.assign(&layer);
        }
    }

    pub fn train(&mut self, data: &[Vec<f64>], expected: &[Vec<f64>], epochs: usize, eta: f64) -> Vec<f64> {
        let mut mses = Vec::with_capacity(epochs);
        
        for epoch in 0..epochs {
            println!("epoch {}/{}", epoch, epochs);
            let mut mse_per_epoch = 0.0;
            for  (x, y) in data.iter().zip(expected.iter()) {
                let mut errors = Vec::with_capacity(self.weights.len());
                let mut weight_deltas = Vec::with_capacity(self.weights.len());
                let mut bias_deltas = Vec::with_capacity(self.biases.len());

                self.feedforward(x);
                let y = Array2::from_shape_vec((y.len(), 1), y.to_vec()).unwrap();
                errors.push(y - self.activations.last().unwrap());

                for weight in self.weights.iter().skip(1).rev() {
                    errors.push(weight.t().dot(errors.last().unwrap()));
                }

                for ((a, o), e) in self
                    .activations
                    .iter()
                    .zip(self.activations.iter().skip(1))
                    .zip(errors.iter().rev())
                {
                    let gradient = e * &o.mapv(&self.activation.derivative) * eta;
                    weight_deltas.push(gradient.dot(&a.t()));
                    bias_deltas.push(gradient);
                }

                for (i, (dw, bw)) in weight_deltas.iter().zip(bias_deltas.iter()).enumerate() {
                    self.weights[i] += dw;
                    self.biases[i] += bw;
                }

                errors
                    .first_mut()
                    .unwrap()
                    .par_mapv_inplace(|n| n.powf(2.0));
                mse_per_epoch += errors.first().unwrap().sum();
            }
            mses.push(mse_per_epoch / data.len() as f64);
        }
        mses
    }
}
