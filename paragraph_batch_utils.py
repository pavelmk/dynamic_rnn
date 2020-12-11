import numpy as np

def batch_data_generator(data, batch_size, n_epochs=-1):
    """ Takes data, then yields it in batches for n_epochs epochs. Wraps around the end
        of the dataset to repeat until the specified number of epochs is reached. We make
        the choice to make all batches equally long (equal to batch_size), rather than terminating
        precisely at the end of the epoch exhaustion.

        inputs:
            data: any-shaped numpy.array. batching is along the 0th dimension.
            batch_size: int >= 1, <= data.shape[0]
            n_epochs: if -1, repeats forever. else, terminates after data is
                      cycled through this many times.

        returns:
            a generator that will repeat the data in the specified batch size.
    """
    if batch_size < 1:
        raise ValueError("Cannot have batch size of < 1: {}".format(batch_size))
    if batch_size > data.shape[0]:
        raise ValueError("Batch size cannot be larger than data! {} > {}".format(batch_size,
                                                                                 data.shape[0]))

    epoch_size = data.shape[0]

    epoch_counter = 0
    idx = 0
    while (epoch_counter < n_epochs) or (n_epochs == -1):
        idx_end = idx + batch_size
        if idx_end < epoch_size:
            yield data[idx:idx_end]
            idx = idx_end
        else:
            # yield through the end then wrap around and grab part of the beginning, too.
            remainder = epoch_size - idx
            yield np.vstack([data[idx:epoch_size], data[0:batch_size-remainder]])
            idx = batch_size-remainder
            epoch_counter += 1


def pad_paragraphs_to_seq_length(paragraph_data, max_seq_length=15000):
    """ Takes the list of paragraphs, pads or truncates them to max_seq_length, and returns an array of padded paragraphs,
        as well as an array of the sequence lengths of those paragraphs.
        inputs:
            paragraph_data: a list of np.arrays of shape [None, 102]. Each list item corresponds to
                         one paragraph. The `None` dimension will be as long as the number of words
                         in a paragraph, and at each word position, the vector is the 102-dim embedding.
            max_seq_length: the length beyond which to truncate, and prior to which to zero-pad.
        returns:
            a tuple of np.arrays of shape [num_paragraphs, max_seq_length, 102] (paragraph data),
            and [num_paragraphs] (seq lengths)
    """
    padded_paragraph_data = []
    sequence_lengths = np.minimum([paragraph.shape[0] for paragraph in paragraph_data], max_seq_length)
    for paragraph in paragraph_data:
        paragraph_length = paragraph.shape[0]
        if paragraph_length < max_seq_length:
            # pad to max seq length
            padded_paragraph = np.pad(paragraph,
                                   [[0, max_seq_length - paragraph.shape[0]], [0,0]],
                                   mode='constant',
                                   constant_values=0)
        elif paragraph_length > max_seq_length:
            # just truncate to max seq length
            padded_paragraph = paragraph[:max_seq_length]
        elif paragraph_length == max_seq_length:
            # perfect length already
            padded_paragraph = paragraph

        padded_paragraph_data.append(padded_paragraph)

    return (np.array(padded_paragraph_data), np.array(sequence_lengths))

