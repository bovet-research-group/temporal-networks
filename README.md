# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/bovet-research-group/temporal-networks/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------ | -------: | -------: | ------: | --------: |
| src/tempnet/logger.py               |       21 |        1 |     95% |        42 |
| src/tempnet/parallel\_expm.py       |      109 |       94 |     14% |39-43, 47-60, 92-122, 127-143, 147, 179-268 |
| src/tempnet/synth\_temp\_network.py |      229 |       70 |     69% |61, 131, 151, 157, 162, 186, 206, 276, 283-285, 297, 325, 329, 335, 339-343, 376, 400, 424, 431-507, 582-584, 595, 636-638 |
| src/tempnet/temporal\_network.py    |      685 |      445 |     35% |174, 184-189, 203-212, 220, 280, 317-353, 391-431, 497-618, 629-686, 703-772, 781-828, 910-922, 988, 995-997, 1002-1004, 1066, 1104-1124, 1221-1293, 1311-1386, 1423-1434, 1466-1489, 1655, 1694-1758, 1765-1772, 1777-1797, 1812-1817, 1834-1879, 1918-1953, 1962, 1973-1991 |
| **TOTAL**                           | **1044** |  **610** | **42%** |           |


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