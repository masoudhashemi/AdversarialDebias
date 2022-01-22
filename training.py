import torch
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm


def pretrain_classifier(
    clf, optimizer_clf, train_loader, loss_criterion, epochs, device
):
    clf.to(device)
    total_classifier_loss = 0
    steps = 0
    pbar = tqdm(range(epochs))
    for _ in pbar:
        epoch_loss = 0
        epoch_batches = 0

        for data in train_loader:

            inputs, label, _ = data
            inputs, label = inputs.to(device), label.to(device)

            optimizer_clf.zero_grad()

            classifier_output = clf(inputs)
            classifier_loss = loss_criterion(classifier_output, label)
            classifier_loss.backward()
            optimizer_clf.step()
            total_classifier_loss += classifier_loss.item()
            epoch_loss += classifier_loss.item()
            epoch_batches += 1
            steps += 1

            pbar.set_description(
                f"Average Clf epoch loss: {epoch_loss/epoch_batches}"
            )

    print("Average Clf batch loss: ", total_classifier_loss / steps)

    return clf


def pretrain_adversary(
    adv, clf, optimizer_adv, train_loader, loss_criterion, epochs, device
):
    adv.to(device)
    total_adversary_loss = 0
    steps = 0
    pbar = tqdm(range(epochs))
    for _ in pbar:
        epoch_loss = 0
        epoch_batches = 0
        for data in train_loader:

            inputs, _, sensitive = data
            inputs, sensitive = inputs.to(device), sensitive.to(device)

            optimizer_adv.zero_grad()

            classifier_output = clf(inputs)
            adversary_output = adv(classifier_output)
            adversary_loss = loss_criterion(adversary_output, sensitive)
            adversary_loss.backward()
            optimizer_adv.step()
            total_adversary_loss += adversary_loss.item()
            epoch_loss += adversary_loss.item()
            epoch_batches += 1
            steps += 1

            pbar.set_description(
                f"Average Adv epoch loss: {epoch_loss/epoch_batches}"
            )

    print("Average Adv batch loss: ", total_adversary_loss / steps)

    return adv


def train_adversary(
    adv, clf, optimizer_adv, train_loader, loss_criterion, epochs, device
):
    adv_loss = 0
    steps = 0
    for _ in range(epochs):
        for data in train_loader:
            inputs, label, sensitive = data
            inputs = inputs.to(device)
            label = label.to(device)
            sensitive = sensitive.to(device)

            optimizer_adv.zero_grad()

            classifier_output = clf(inputs)
            adversary_output = adv(classifier_output)
            adversary_loss = loss_criterion(adversary_output, sensitive)
            adversary_loss.backward()
            optimizer_adv.step()
            adv_loss += adversary_loss.item()
            steps += 1

    return adv


def train_classifier(
    clf, optimizer_clf, adv, data, loss_criterion, lbda, device
):
    inputs, label, sensitive = data
    inputs = inputs.to(device)
    label = label.to(device)
    sensitive = sensitive.to(device)

    optimizer_clf.zero_grad()

    classifier_output = clf(inputs)
    adversary_output = adv(classifier_output)
    adversary_loss = loss_criterion(adversary_output, sensitive)
    classifier_loss = loss_criterion(classifier_output, label)
    total_classifier_loss = classifier_loss - lbda * adversary_loss
    total_classifier_loss.backward()

    optimizer_clf.step()

    print("Adversary Mini-Batch loss: ", adversary_loss.item())
    print("Classifier Mini-Batch loss: ", classifier_loss.item())
    print("Total Mini-Batch loss: ", total_classifier_loss.item())

    return clf


def inf_loader(loader):
    while True:
        for data in loader:
            yield data


def train_debiased(
    clf, adv, train_loader, lbda, iterations, epochs_clf, epochs_adv, device
):

    loss_criterion = torch.nn.CrossEntropyLoss()
    gr_train_loader = inf_loader(train_loader)
    # Defining optimizers
    optimizer_adv = optim.Adam(adv.parameters(), lr=0.001)
    optimizer_clf = optim.Adam(clf.parameters(), lr=0.001)

    clf.to(device)
    adv.to(device)

    # PRETRAIN CLASSIFIER

    for param in adv.parameters():
        param.requires_grad = False

    clf = pretrain_classifier(
        clf, optimizer_clf, train_loader, loss_criterion, epochs_clf, device
    )

    for param in adv.parameters():
        param.requires_grad = True

    # PRETRAIN ADVERSARY

    for param in clf.parameters():
        param.requires_grad = False

    adv = pretrain_adversary(
        adv,
        clf,
        optimizer_adv,
        train_loader,
        loss_criterion,
        epochs_adv,
        device,
    )

    for param in clf.parameters():
        param.requires_grad = True

    print("Lambda: " + str(lbda))

    for iteration in range(iterations):  # loop over the dataset multiple times
        print("Iteration: ", iteration)

        # TRAIN ADVERSARY FOR 1 EPOCH

        for param in clf.parameters():
            param.requires_grad = False

        adv = train_adversary(
            adv,
            clf,
            optimizer_adv,
            train_loader,
            loss_criterion,
            epochs=1,
            device=device,
        )

        for param in clf.parameters():
            param.requires_grad = True

        # TRAIN CLASSIFIER FOR 1 EPOCH

        for param in adv.parameters():
            param.requires_grad = False

        data = next(gr_train_loader)
        clf = train_classifier(
            clf, optimizer_clf, adv, data, loss_criterion, lbda, device
        )

        for param in adv.parameters():
            param.requires_grad = True

    return clf, adv
