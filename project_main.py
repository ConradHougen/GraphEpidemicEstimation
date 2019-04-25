# Run experiments for project

from project_utils import *
import numpy as np
import matplotlib.pyplot as plt

contact_filenames = ['contact/tij_InVS13.dat',
                    'contact/tij_InVS15.dat',
                    'contact/tij_LH10.dat',
                    'contact/tij_LyonSchool.dat',
                    'contact/tij_SFHH.dat',
                    'contact/tij_Thiers13.dat']

copresence_filenames = ['co-presence/tij_pres_InVS13.dat',
                       'co-presence/tij_pres_InVS15.dat',
                       'co-presence/tij_pres_LH10.dat',
                       'co-presence/tij_pres_LyonSchool.dat',
                       'co-presence/tij_pres_SFHH.dat',
                       'co-presence/tij_pres_Thiers13.dat']

metadata_filenames = ['metadata/metadata_InVS13.dat',
                     'metadata/metadata_InVS15.dat',
                     'metadata/metadata_LH10.dat',
                     'metadata/metadata_LyonSchool.dat',
                     'metadata/metadata_SFHH.dat',
                     'metadata/metadata_Thiers13.dat']

contact_output_filenames = ['output/temporally_aggregate_matrices/contact/agg_mat_InVS13.npy',
                           'output/temporally_aggregate_matrices/contact/agg_mat_InVS15.npy',
                           'output/temporally_aggregate_matrices/contact/agg_mat_LH10.npy',
                           'output/temporally_aggregate_matrices/contact/agg_mat_LyonSchool.npy',
                           'output/temporally_aggregate_matrices/contact/agg_mat_SFHH.npy',
                           'output/temporally_aggregate_matrices/contact/agg_mat_Thiers13.npy']

copresence_output_filenames = ['output/temporally_aggregate_matrices/co-presence/agg_mat_pres_InVS13.npy',
                               'output/temporally_aggregate_matrices/co-presence/agg_mat_pres_InVS15.npy',
                               'output/temporally_aggregate_matrices/co-presence/agg_mat_pres_LH10.npy',
                               'output/temporally_aggregate_matrices/co-presence/agg_mat_pres_LyonSchool.npy',
                               'output/temporally_aggregate_matrices/co-presence/agg_mat_pres_SFHH.npy',
                               'output/temporally_aggregate_matrices/co-presence/agg_mat_pres_Thiers13.npy']

def plot_k_largest_eigvals_of_contact_and_copresence_matrices():
    NUM_EIGVALS = 10

    for f_idx in range(0, 6):
        A_contact = np.load(contact_output_filenames[f_idx])
        A_pres = np.load(copresence_output_filenames[f_idx])

        n = A_contact.shape[0]

        deg_vec_contact = np.sum(A_contact, axis=1)
        deg_vec_pres = np.sum(A_pres, axis=1)
        inv_sqrt_vec_contact = np.zeros(deg_vec_contact.shape)
        inv_sqrt_vec_pres = np.zeros(deg_vec_pres.shape)
        for i in range(0, n):
            if deg_vec_contact[i] != 0:
                inv_sqrt_vec_contact[i] = 1 / np.sqrt(deg_vec_contact[i])
            if deg_vec_pres[i] != 0:
                inv_sqrt_vec_pres[i] = 1 / np.sqrt(deg_vec_pres[i])

        D_contact = np.diag(inv_sqrt_vec_contact)
        D_pres = np.diag(inv_sqrt_vec_pres)

        # A_contact[A_contact > 0] = 1
        # A_pres[A_pres > 0] = 1

        A_contact_norm = np.matmul(np.matmul(D_contact, A_contact), D_contact)
        A_pres_norm = np.matmul(np.matmul(D_pres, A_pres), D_pres)

        plot_k_largest_eigvals_of_two_matrices(A_contact_norm, A_pres_norm, NUM_EIGVALS)


def save_and_plot_all_temporally_aggregated_adjacency_matrices():
    for f_idx in range(0, 6):
        nodes = create_node_df_from_metadata_file(metadata_filenames[f_idx])

        plt.figure(f_idx + 1)

        tij = create_tij_df_from_data_file(contact_filenames[f_idx])
        A = create_temporally_aggregated_adj_mat(tij, nodes)
        np.save(contact_output_filenames[f_idx], A)
        print("Saved {}".format(contact_output_filenames[f_idx]))
        plt.subplot(1, 2, 1)
        plt.matshow(A, fignum=False)

        tij = create_tij_df_from_data_file(copresence_filenames[f_idx])
        A = create_temporally_aggregated_adj_mat(tij, nodes)
        np.save(copresence_output_filenames[f_idx], A)
        print("Saved {}".format(copresence_output_filenames[f_idx]))
        plt.subplot(1, 2, 2)
        plt.matshow(A, fignum=False)

        plt.show()


if __name__ == "__main__":
    print("Running current enabled project experiments...")
    # plot_k_largest_eigvals_of_contact_and_copresence_matrices()
    # save_and_plot_all_temporally_aggregated_adjacency_matrices()

