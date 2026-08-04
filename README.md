# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/bovet-research-group/temporal-networks/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------ | -------: | -------: | ------: | --------: |
| src/tempnet/logger.py               |       21 |        1 |     95% |        42 |
| src/tempnet/parallel\_expm.py       |      111 |       94 |     15% |42-46, 50-63, 95-125, 130-146, 150, 182-271 |
| src/tempnet/synth\_temp\_network.py |      231 |       70 |     70% |65, 135, 155, 161, 166, 190, 210, 280, 287-289, 301, 329, 333, 339, 343-347, 380, 404, 428, 435-511, 586-588, 599, 640-642 |
| src/tempnet/temporal\_network.py    |      629 |      384 |     39% |180, 190-195, 209-218, 231, 298, 335-371, 409-449, 515-632, 643-700, 717-785, 794-841, 923-935, 986, 1013, 1020-1022, 1027-1029, 1098, 1136-1156, 1253-1325, 1343-1418, 1455-1466, 1468, 1499-1519, 1691, 1730-1794, 1833-1868 |
| src/tempnet/utils.py                |       55 |       43 |     22% |62, 79-87, 116-122, 168-191, 213-219, 238-257 |
| **TOTAL**                           | **1047** |  **592** | **43%** |           |


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