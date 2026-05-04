# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/bovet-research-group/temporal-networks/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------ | -------: | -------: | ------: | --------: |
| src/tempnet/logger.py               |       21 |        1 |     95% |        42 |
| src/tempnet/parallel\_expm.py       |      109 |       94 |     14% |39-43, 47-60, 92-122, 127-143, 147, 179-268 |
| src/tempnet/synth\_temp\_network.py |      276 |      116 |     58% |63, 133, 153, 159, 164, 188, 208, 278, 285-287, 299, 327, 331, 337, 341-345, 378, 402, 426, 433-509, 584-586, 597, 638-640, 647-731 |
| src/tempnet/temporal\_network.py    |      990 |      740 |     25% |174, 184-189, 203-212, 220, 280, 321-361, 403-447, 513-634, 649-774, 785-875, 892-961, 977-1052, 1062-1131, 1213-1225, 1291, 1298-1300, 1305-1307, 1369, 1407-1427, 1524-1596, 1626-1722, 1739-1814, 1830-1904, 1913-1945, 1980-1991, 2023-2076, 2242, 2279, 2324-2344, 2362-2398, 2433-2497, 2504-2511, 2516-2536, 2551-2556, 2573-2618, 2657-2692, 2727-2745, 2761-2765, 2774, 2785-2803 |
| **TOTAL**                           | **1396** |  **951** | **32%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/bovet-research-group/temporal-networks/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/bovet-research-group/temporal-networks/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/bovet-research-group/temporal-networks/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/bovet-research-group/temporal-networks/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fbovet-research-group%2Ftemporal-networks%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/bovet-research-group/temporal-networks/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.