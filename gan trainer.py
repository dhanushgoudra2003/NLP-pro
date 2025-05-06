epochs = 800
batch_size = 1024
gen_losses, disc_losses = [], []

for epoch in range(epochs):
    perm = torch.randperm(len(X_train))
    X_train = X_train[perm]

    for i in range(0, len(X_train), batch_size):
        real_batch = X_train[i:i+batch_size].to(device)
        noise = torch.randn(real_batch.size(0), input_dim).to(device)
        fake_batch = generator(noise)

        # Label smoothing
        real_labels = torch.empty((real_batch.size(0), 1), device=device).uniform_(0.8, 1.0)
        fake_labels = torch.zeros((real_batch.size(0), 1), device=device)

        # Random label flipping
        if np.random.rand() < 0.05:
            real_labels, fake_labels = fake_labels, real_labels

        # Train Discriminator
        disc_optimizer.zero_grad()
        real_loss = loss_fn(discriminator(real_batch), real_labels)
        fake_loss = loss_fn(discriminator(fake_batch.detach()), fake_labels)
        disc_loss = real_loss + fake_loss
        disc_loss.backward()
        disc_optimizer.step()

        # Train Generator
        gen_optimizer.zero_grad()
        gen_loss = loss_fn(discriminator(fake_batch), torch.ones_like(real_labels))  # fool the discriminator
        penalty = torch.mean((fake_batch.mean(dim=0) - real_batch.mean(dim=0))**2)
        total_gen_loss = gen_loss + 0.1 * penalty
        total_gen_loss.backward()
        gen_optimizer.step()

        gen_losses.append(gen_loss.item())
        disc_losses.append(disc_loss.item())

    print(f"Epoch {epoch+1}/{epochs} - Gen Loss: {gen_loss.item():.4f}, Disc Loss: {disc_loss.item():.4f}")
