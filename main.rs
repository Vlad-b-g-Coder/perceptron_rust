use tch::{nn, nn::Module, nn::OptimizerConfig, Tensor, Device};
use std::io;

fn generate_data(samples: usize) -> (Tensor, Tensor) {
    let x = Tensor::randn(&[samples as i64, 1], (tch::Kind::Float, Device::Cpu));
    let y = 2.0 * x.pow(&Tensor::from_slice(&[5.0]).to_kind(tch::Kind::Float))
        + 1.0
        + Tensor::randn(&[samples as i64, 1], (tch::Kind::Float, Device::Cpu)) * 0.1;
    (x, y)
}

fn main() -> tch::Result<()> {
    let device = Device::cuda_if_available();
    let lr = 1e-4;
    let epochs = 4000;
    let batch_size = 32;

    let (x_train, y_train) = generate_data(1000);
    let (x_val, y_val) = generate_data(200);

    let vs = nn::VarStore::new(device);
    let net = nn::seq()
        .add(nn::linear(&vs.root() / "layer1", 1, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&vs.root() / "layer2", 64, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&vs.root() / "layer3", 64, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&vs.root() / "output", 64, 1, Default::default()));

    let mut opt = nn::Adam::default().build(&vs, lr)?;

    for epoch in 1..=epochs {
        let indices = Tensor::randperm(x_train.size()[0], (tch::Kind::Int64, device));
        let batches = indices.split(batch_size as i64, 0);
        for batch in batches {
            let xb = x_train.index_select(0, &batch).to_device(device);
            let yb = y_train.index_select(0, &batch).to_device(device);

            let loss = net.forward(&xb).mse_loss(&yb, tch::Reduction::Mean);
            opt.backward_step(&loss);
        }

        if epoch % 500 == 0 {
            let val_loss = net
                .forward(&x_val.to_device(device))
                .mse_loss(&y_val.to_device(device), tch::Reduction::Mean);
            println!("Epoch: {:4}, Val Loss: {:.4}", epoch, val_loss.double_value(&[]));
        }
    }

    loop {
        println!("Enter x value (or type 'exit' to quit):");
        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Failed to read input");
        let input = input.trim();

        if input.eq_ignore_ascii_case("exit") {
            break;
        }

        let x_value: f32 = match input.parse() {
            Ok(v) => v,
            Err(_) => {
                println!("Invalid input. Please enter a valid number.");
                continue;
            }
        };

        let x_tensor = Tensor::from_slice(&[x_value]).to_kind(tch::Kind::Float).unsqueeze(0);
        let y_pred = net.forward(&x_tensor.to_device(device)).to_device(Device::Cpu);
        let y_true = 2.0 * x_tensor.pow(&Tensor::from(5.0)) + 1.0;

        let predicted_value = y_pred;
        let true_value_scalar = y_true;

        println!("x value: {}", x_value);
        println!("Predicted y: {}", predicted_value);
        println!("True y (formula): {}", true_value_scalar);
    }

    Ok(())
}
