# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/bovet-research-group/temporal-networks/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------ | -------: | -------: | ------: | --------: |
| src/tempnet/expm\_with\_tol.py      |      122 |       15 |     88% |195-221, 258 |
| src/tempnet/faster\_expm.py         |      143 |       10 |     93% |32-36, 41-54, 123 |
| src/tempnet/logger.py               |       21 |        1 |     95% |        74 |
| src/tempnet/sanitize.py             |       67 |        0 |    100% |           |
| src/tempnet/synth\_temp\_network.py |      228 |       67 |     71% |53, 123, 143, 149, 154, 178, 198, 268, 275-277, 289, 317, 321, 327, 331-335, 368, 392, 416, 423-499, 583, 623-625 |
| src/tempnet/temporal\_network.py    |      737 |      249 |     66% |201-202, 224, 306, 538-655, 666-723, 760, 764-772, 780, 786, 794-800, 804, 821-831, 834-835, 855-856, 910, 983, 987-990, 1047, 1074, 1165, 1220-1227, 1359, 1368-1370, 1376-1379, 1389, 1394, 1405-1409, 1412-1413, 1442, 1446-1447, 1451, 1461-1462, 1482, 1498, 1560-1591, 1605-1684, 1732, 1768-1788, 1978, 2067, 2077, 2081-2084 |
| src/tempnet/utils.py                |       73 |       16 |     78% |46, 155, 169, 175, 286, 291, 310-329 |
| **TOTAL**                           | **1391** |  **358** | **74%** |           |


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