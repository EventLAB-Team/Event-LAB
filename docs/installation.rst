Installation
============

Packages and dependencies
-------------------------

Event-LAB is powered by `Pixi <https://pixi.prefix.dev/latest/>`_ for all package and dependency management. This allows
Event-LAB to handle all the baseline requirements and allows it to be cross platform for seamless integration. If you have
not already installed Pixi, run the following command in your teminal:

.. code-block:: bash
    
    curl -fsSL https://pixi.sh/install.sh | bash

The provided ``pixi.toml`` and ``pixi.lock`` files in the repository handle all packages and dependencies and the operation of all the various functions in LENS.
This documentation focuses on using pixi to run evaluation and training scripts.

.. note::

    You will be prompted to restart your terminal after installation, please do so before proceeding. For more information, 
    visit the `pixi documentation <https://pixi.prefix.dev/latest/>`_.

Clone the repository
--------------------

Next, the only thing you need to do is clone the repository and navigate to the project directory:

.. code-block:: bash

   git clone git@github.com:EventLAB-Team/Event-LAB.git && cd Event-LAB

Check installation
--------------------

To check the installation, simply run the following command in your terminal:

.. code-block:: bash

   pixi run demo

This will trigger the installation of the necessary packages from the ``pixi.lock`` file and then run a small
demo script. If it is successful, you will see a Recall@K table printed in your terminal.