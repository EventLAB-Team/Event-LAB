Quick start
===========

To get started using Event-LAB, the general format for running the code is as follows:

.. code-block:: bash

   pixi run eventlab <baseline> <dataset> <reference> <query>

This basic formula is used to combine any baseline with any dataset reference and query pair you wish to run. 

Example
-------

.. code-block:: bash

   pixi run eventlab sparse_event brisbane_event sunset1 sunset2

In the above example, we will run the ``sparse_event`` baseline on the ``brisbane_event`` dataset, using ``sunset1`` as the 
reference and ``sunset2`` as the query. 

Event-LAB triggers multiple steps to clone the specified baseline, download the reference and query source dataset,
and generate the event data based on the user specified configuration file. 

Once complete, the baseline will run using the reference and query pair generated and calculate a Recall@K metric
and a precision-recall curve.