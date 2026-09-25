Here is a summary of the final data structure being returned from the [GraphDataset](cci:2://file:///Users/xjin/my_home/git/ocelot3/ocelot/graph_dataset.py:10:0-154:19):

*   **Input Nodes**:
    *   For **satellite data**, rows with invalid features are completely removed.
    *   For **conventional data**, a `valid_mask` is concatenated with the input features.
*   **Target Nodes**:
    *   The ground truth features are stored in `data[...].y`.
    *   A corresponding `data[...].valid_mask` is now included. This boolean tensor has the same shape as `y` and indicates which of the target values are valid and should be used when calculating the loss.

This completes the data preparation pipeline. Your model now receives clean, well-structured data with all the necessary validity information for both its inputs and its targets.

