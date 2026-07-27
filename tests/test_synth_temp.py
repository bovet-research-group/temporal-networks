"""
#
# Temporal networks `tempnet`
#
# Copyright (C) 2021 Alexandre Bovet <alexandre.bovet@uzh.ch>
# Copyright (C) 2026 Alexandre Bovet <alexandre.bovet@uzh.ch>, 
#                    Yasaman Asgari <yasaman.asgari@uzh.ch>, 
#                    Samuel Koovely <samuel.koovely@uzh.ch>, 
#                    Jonas Liechti <jonas@t4d.ch>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


"""


import pytest


def test_distro():
    from tempnet.synth_temp_network import (
        Distro,
    )
    d = Distro(loc=0.0, scale=1.0)
    _ = d.draw_val()
    _ = d.draw_val(loc=0.5)
    _ = d.draw_val(scale=0.5)
    _ = d.draw_val(loc=0.5, scale=1.5)


# you are here!
@pytest.mark.parametrize(
        "_id, i_d_loc, i_d_scale, i_d_mf, a_d_loc, a_d_scale, a_d_mf,"
        " dist_type, group",
        [(1, 0.0, 1.0, None, 0.0, 1.0, None, "exponential", 0),]
    )
def test_individual(_id,
                    i_d_loc, i_d_scale, i_d_mf,
                    a_d_loc, a_d_scale, a_d_mf,
                    dist_type, group):
    from tempnet.synth_temp_network import (
        Individual,
    )
    i1 = Individual(ID=_id,
                    inter_distro_loc=i_d_loc,
                    inter_distro_scale=i_d_scale,
                    inter_distro_type=dist_type,
                    inter_distro_mod_func=i_d_mf,
                    activ_distro_loc=a_d_loc,
                    activ_distro_scale=a_d_scale,
                    activ_distro_type=dist_type,
                    activ_distro_mod_func=a_d_mf,
                    group=group)
    _ = i1.draw_inter_duration(time=None)
    _ = i1.draw_inter_duration(time=1.0)
    _ = i1.draw_activ_time(time=None)
    _ = i1.draw_activ_time(time=1.0)


def test_synth_temp_network():
    from tempnet.synth_temp_network import (
        Individual,
        SynthTempNetwork
    )
    individuals = [Individual(i, group=0) for i in range(20)]
    sim = SynthTempNetwork(individuals, t_start=0, t_end=50)

    sim.run(save_all_states=True, save_dt_states=True, verbose=True)
