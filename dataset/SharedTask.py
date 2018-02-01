# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset
import unicodecsv
from cStringIO import StringIO
import io
import codecs


# Shared Task 2018
class SharedTaskDataset(Dataset):
    """
    Shared Task 2018
    """

    # Constructor
    def __init__(self, csv_file, train=True, k=20):
        """
        Constructor
        :param csv_file: Path to CSV file
        """
        # Load samples
        self.samples, self.labels = self._load_samples(csv_file, train, k)
    # end __init__

    #####################################
    # OVERRIDE
    #####################################

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return len(self.samples)
    # end __len__

    # Get item
    def __getitem__(self, item):
        """
        Get item
        :param item:
        :return:
        """
        return self.samples[item], self.labels[item]
    # end __getitem__

    ########################################
    # PRIVATE
    ########################################

    # Load samples
    def _load_samples(self, csv_file, train=True, k=20):
        """
        Load samples
        :param csv_file:
        :return:
        """
        # Training length
        training_length = int(27299 / k) * (k - 1)

        # Empty samples and labels
        samples = list()
        labels = list()

        # CSV reader
        r = unicodecsv.reader(codecs.open(csv_file, 'rb', encoding='utf-8'))

        # Read
        for index, row in enumerate(r):
            if (index < training_length and train) or (index >= training_length and not train):
                # Features
                features = torch.zeros(1, 416)

                # For each feature
                for i in range(416):
                    features[0, i] = float(row[i])
                # end for

                # Label
                label = int(row[-1])

                # Add
                samples.append(features)
                labels.append(label)
            # end if
        # end for

        return samples, labels
    # end _load_samples

# end SharedTaskDataset
