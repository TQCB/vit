import torch

def checkpoint(model, optimizer, filename):
    torch.save(
        {'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}, # meli likes you
        filename
        )

def resume(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def train_network(network, trainloader, optimizer, criterion, epochs, checkpoint_interval=None, checkpoint_path=None, print_interval=100, n_checkpoint=0):
    if n_checkpoint > 0:
        resume_cp_path = checkpoint_path + str(n_checkpoint) + '.pkl'
        resume(network, optimizer, resume_cp_path)

    for epoch in range(epochs):
        running_loss = 0.0
        total_correct = 0.0
        total_samples = 0.0

        for i, data in enumerate(trainloader, 0):
            # get inoputs
            inputs, labels = data

            # zero param gradients
            optimizer.zero_grad()

            # forward -> backward -> optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # metric
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # statistics
            running_loss += loss.item()

            # Print stats at print interval
            if i % print_interval == print_interval-1:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_interval:.3f}, accuracy: {100 * total_correct / total_samples:.1f}%')
                running_loss = 0.0

            # Checkpoint model at checkpoint interval
            if checkpoint_interval:
                if i % checkpoint_interval == checkpoint_interval-1:
                    n_checkpoint += 1
                    cp_path = checkpoint_path + str(n_checkpoint) + '.pkl'
                    checkpoint(network, optimizer, cp_path)

    print("Training Finished")