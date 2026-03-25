from src.tempnet import temporal_network as tn
import pandas as pd
import numpy as np
base_folder='\\temporal-networks\\Tests\\Prepare Real test\\'
url = 'https://zenodo.org/record/4725155/files/'\
                        'mice_contact_sequence.csv.gz'
cut_after = 24*3600 

raw_df = pd.read_csv(url,compression='gzip')
raw_df = raw_df[raw_df['ending_times'] < cut_after]
    
source_nodes = raw_df['source_nodes'].tolist()
target_nodes = raw_df['target_nodes'].tolist()
starting_times = raw_df['starting_times'].round(3).tolist()
ending_times = raw_df['ending_times'].round(3).tolist()
cont_net=tn.ContTempNetwork(\
    source_nodes=source_nodes,\
    target_nodes=target_nodes,\
    starting_times=starting_times,\
    ending_times=ending_times,\
    relabel_nodes=True
)
cont_net._compute_time_grid()
A=cont_net.compute_static_adjacency_matrix()
A_1h=cont_net.compute_static_adjacency_matrix(start_time=0, end_time=3600)
np.save('Tests/Prepare Real test/mice_node_array.npy', cont_net.node_array)
np.save('Tests/Prepare Real test/mice_full_adjacency.npy', A.toarray())
np.save('Tests/Prepare Real test/mice_1h_adjacency.npy', A_1h.toarray())
np.save('Tests/Prepare Real test/mice_times.npy', cont_net.times)
cont_net.events_table.to_csv('Tests/Prepare Real test/mice_event_table.csv', index=False)
cont_net.time_grid.reset_index().to_csv('Tests/Prepare Real test/mice_time_grid.csv', index=False)