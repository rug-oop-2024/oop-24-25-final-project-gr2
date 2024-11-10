### AutoOOP

The **model class** is an abstract base class that is the blueprint for the implemented regression and classification models.

For the **regression models** I chose Lasso, Gradient Boosting, MLR and Random Forest after looking at the models from sklearn, and these seemed fitting for the assignment at hand.

Similarly, for the **classification models** I chose Gradient Boosting Classification, Random Forest and Logistic Regression, again wrapping the sklearn models. 
 
The models use default parameters by default.


In the **artifact** I implemented properties for encapsulation, making sure to return deepcopies of mutable objects.
The artifact can be converted to a dictionary, and a class method that returns an artifact from a given dictionary.

In the **dataset class** there are methods for reading and saving an artifact, and creating a dataset from a pandas dataframe. As the dataset is an artifact, it inherits it's attributes from the artifact class.

In the **feature class** I implemented properties for encapsulation, and overloaded the equality.

The **metrics** file includes the metric base class, metrics for regression, and metrics for classification. The metrics make use of the `__call__` function. For both the regression and classification metrics I looked at metrics from sklearn for their specific task, and implemented the ones that seemed fit for the project. For the regression metrics, I opted for removing the square log error which i initially chose instead of max error, as it is possible that the ground truth or predictions are negative, so I considered it is better to avoid unneccessary issues. For the classification metrics, I chose to implement a confusion matrix class as well (computes the matrix, and has an abstract method for calculating the metric to be implemented in any metric class that needs to use a confusion matrix, it also calls the method for computing the metric in `__call__`). The metrics that were calculated using a confusion matrix inherit from the aformentioned class, therefore their call method is inherited, so they implement the compute metric. For each of the metric in the constructor their name is set, for later easier access.

I extended the **pipeline** class by adding a private method for evaluating the training data. And extended the execute function for the training data, also making sure to flatten the arrays, and decode them from the OneHotEncoding. 

### Functional

In feature.py I find the type of features in the dataset and make two categories : numerical or categorical. The function returns a list of features with name and type.

### Tests 

Small adjustments were made in the tests for correct imports and naming as per my implementation, this was done without tampering with the functionality.


### App

In the **datasets page** the uploading of cvs files is handled. The data is showcased in a table after uploading.
I enforce only accepting csv format.
I also added a button to download the dataset to make it "easier" for grading.

In the **modelling page** the user can choose their own configuration for dataset, feature and model selection, splitting the data and choosing metrics. Messages are shown to guide the user on what is requested, and the interface is straightforward. The user can create a pipeline, then see it's summary. The user can also train the model if they agree with the summary. Additionally the user can save the pipeline. 

### Extra Requirenments 

I also implemented the page **deployment**, where the existing pipelines can be seen from a menu and selected. Upon selection a summary of the pipeline will be displayed.

### Notes 

There are datasets already saved in the app, that can be used to test the functionality.