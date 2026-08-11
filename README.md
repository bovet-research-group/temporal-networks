# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/bovet-research-group/temporal-networks/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------ | -------: | -------: | ------: | --------: |
| src/tempnet/expm\_with\_tol.py      |      123 |       15 |     88% |213-239, 276 |
| src/tempnet/faster\_expm.py         |      143 |       10 |     93% |45-49, 54-67, 136 |
| src/tempnet/logger.py               |       22 |        1 |     95% |        93 |
| src/tempnet/sanitize.py             |       68 |        0 |    100% |           |
| src/tempnet/synth\_temp\_network.py |      228 |       67 |     71% |69, 139, 159, 165, 170, 194, 214, 284, 291-293, 305, 333, 337, 343, 347-351, 384, 408, 432, 439-515, 599, 639-641 |
| src/tempnet/temporal\_network.py    |      737 |      249 |     66% |219-220, 242, 324, 556-673, 684-741, 778, 782-790, 798, 804, 812-818, 822, 839-849, 852-853, 873-874, 928, 1001, 1005-1008, 1065, 1092, 1183, 1238-1245, 1377, 1386-1388, 1394-1397, 1407, 1412, 1423-1427, 1430-1431, 1460, 1464-1465, 1469, 1479-1480, 1500, 1516, 1578-1609, 1623-1702, 1750, 1786-1806, 1996, 2085, 2095, 2099-2102 |
| src/tempnet/utils.py                |       74 |       16 |     78% |66, 175, 189, 195, 306, 311, 330-349 |
| **TOTAL**                           | **1395** |  **358** | **74%** |           |


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