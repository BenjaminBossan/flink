# FLINK

A package containing Online Gradient Descent Logistic Regression (OGDLR) at the moment.

The implementation uses dictionaries (OGDLRd) or arrays via hashing trick (OGDLRa) to
be as sparse as possible. All training values must be converted to string and are thus
treated categorically. This makes for an extremely flexible logistic regression that
does not require prior knowledge about the incoming data.

This is an extremely early version, use at your own risk. More features 
such as regularization will come soon. For example usage, see the functional tests.