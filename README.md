# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/bovet-research-group/temporal-networks/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------ | -------: | -------: | ------: | --------: |
| src/tempnet/faster\_expm.py         |      143 |       10 |     93% |45-49, 54-67, 136 |
| src/tempnet/logger.py               |       21 |        1 |     95% |        42 |
| src/tempnet/synth\_temp\_network.py |      228 |       67 |     71% |65, 135, 155, 161, 166, 190, 210, 280, 287-289, 301, 329, 333, 339, 343-347, 380, 404, 428, 435-511, 595, 635-637 |
| src/tempnet/temporal\_network.py    |      594 |      304 |     49% |175, 185-190, 204-213, 226, 514-631, 642-699, 716-784, 793-840, 922-934, 985, 1012, 1019-1021, 1026-1028, 1097, 1135-1155, 1252-1324, 1342-1417, 1454-1465, 1467, 1498-1518, 1792 |
| src/tempnet/utils.py                |       55 |       34 |     38% |62, 79-87, 168-191, 214, 219, 238-257 |
| **TOTAL**                           | **1041** |  **416** | **60%** |           |


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