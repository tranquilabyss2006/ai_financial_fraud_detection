"""
Graph Features Module
Implements graph-based feature extraction techniques for fraud detection
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
import community as community_louvain
from sklearn.preprocessing import MinMaxScaler
import warnings
import logging
from typing import Dict, List, Tuple, Union

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphFeatures:
    """
    Class for extracting graph-based features from transaction data
    Implements techniques like centrality measures, clustering coefficients, etc.
    """
    
    def __init__(self, config=None):
        """
        Initialize GraphFeatures
        
        Args:
            config (dict, optional): Configuration parameters
        """
        self.config = config or {}
        self.graph = None
        self.sender_graph = None
        self.receiver_graph = None
        self.bipartite_graph = None
        self.feature_names = []
        self.scaler = MinMaxScaler()
        self.fitted = False
        
    def extract_features(self, df):
        """
        Extract all graph features from the dataframe
        
        Args:
            df (DataFrame): Input transaction data
            
        Returns:
            DataFrame: DataFrame with extracted features
        """
        try:
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Build graphs
            self._build_graphs(df)
            
            # Extract different types of graph features
            result_df = self._extract_centrality_features(result_df)
            result_df = self._extract_clustering_features(result_df)
            result_df = self._extract_community_features(result_df)
            result_df = self._extract_path_features(result_df)
            result_df = self._extract_subgraph_features(result_df)
            result_df = self._extract_temporal_graph_features(result_df)
            
            # Store feature names
            self.feature_names = [col for col in result_df.columns if col not in df.columns]
            
            logger.info(f"Extracted {len(self.feature_names)} graph features")
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting graph features: {str(e)}")
            raise
    
    def _build_graphs(self, df):
        """
        Build various graphs from the transaction data
        
        Args:
            df (DataFrame): Input transaction data
        """
        try:
            # Build transaction graph (directed)
            self.graph = nx.DiGraph()
            
            # Add edges with attributes
            for _, row in df.iterrows():
                if 'sender_id' in row and 'receiver_id' in row:
                    sender = row['sender_id']
                    receiver = row['receiver_id']
                    
                    # Get edge attributes
                    attrs = {}
                    if 'amount' in row:
                        attrs['amount'] = row['amount']
                    if 'timestamp' in row:
                        attrs['timestamp'] = row['timestamp']
                    if 'transaction_id' in row:
                        attrs['transaction_id'] = row['transaction_id']
                    
                    # Add edge or update existing edge
                    if self.graph.has_edge(sender, receiver):
                        # Update existing edge
                        edge_data = self.graph[sender][receiver]
                        if 'amount' in attrs:
                            edge_data['total_amount'] = edge_data.get('total_amount', 0) + attrs['amount']
                        edge_data['transaction_count'] = edge_data.get('transaction_count', 0) + 1
                        edge_data['transactions'].append(attrs)
                    else:
                        # Add new edge
                        attrs['total_amount'] = attrs.get('amount', 0)
                        attrs['transaction_count'] = 1
                        attrs['transactions'] = [attrs]
                        self.graph.add_edge(sender, receiver, **attrs)
            
            # Build sender graph (undirected, senders connected if they have common receivers)
            self.sender_graph = nx.Graph()
            
            # Create a mapping from receivers to senders
            receiver_to_senders = defaultdict(set)
            for _, row in df.iterrows():
                if 'sender_id' in row and 'receiver_id' in row:
                    receiver_to_senders[row['receiver_id']].add(row['sender_id'])
            
            # Connect senders with common receivers
            for receiver, senders in receiver_to_senders.items():
                senders_list = list(senders)
                for i in range(len(senders_list)):
                    for j in range(i+1, len(senders_list)):
                        sender1 = senders_list[i]
                        sender2 = senders_list[j]
                        
                        if self.sender_graph.has_edge(sender1, sender2):
                            self.sender_graph[sender1][sender2]['common_receivers'] += 1
                        else:
                            self.sender_graph.add_edge(sender1, sender2, common_receivers=1)
            
            # Build receiver graph (undirected, receivers connected if they have common senders)
            self.receiver_graph = nx.Graph()
            
            # Create a mapping from senders to receivers
            sender_to_receivers = defaultdict(set)
            for _, row in df.iterrows():
                if 'sender_id' in row and 'receiver_id' in row:
                    sender_to_receivers[row['sender_id']].add(row['receiver_id'])
            
            # Connect receivers with common senders
            for sender, receivers in sender_to_receivers.items():
                receivers_list = list(receivers)
                for i in range(len(receivers_list)):
                    for j in range(i+1, len(receivers_list)):
                        receiver1 = receivers_list[i]
                        receiver2 = receivers_list[j]
                        
                        if self.receiver_graph.has_edge(receiver1, receiver2):
                            self.receiver_graph[receiver1][receiver2]['common_senders'] += 1
                        else:
                            self.receiver_graph.add_edge(receiver1, receiver2, common_senders=1)
            
            # Build bipartite graph (senders and receivers as two separate sets)
            self.bipartite_graph = nx.Graph()
            
            # Add nodes with bipartite attribute
            for _, row in df.iterrows():
                if 'sender_id' in row:
                    self.bipartite_graph.add_node(row['sender_id'], bipartite=0)
                if 'receiver_id' in row:
                    self.bipartite_graph.add_node(row['receiver_id'], bipartite=1)
            
            # Add edges
            for _, row in df.iterrows():
                if 'sender_id' in row and 'receiver_id' in row:
                    sender = row['sender_id']
                    receiver = row['receiver_id']
                    
                    # Get edge attributes
                    attrs = {}
                    if 'amount' in row:
                        attrs['amount'] = row['amount']
                    if 'timestamp' in row:
                        attrs['timestamp'] = row['timestamp']
                    if 'transaction_id' in row:
                        attrs['transaction_id'] = row['transaction_id']
                    
                    # Add edge
                    self.bipartite_graph.add_edge(sender, receiver, **attrs)
            
            logger.info("Graphs built successfully")
            
        except Exception as e:
            logger.error(f"Error building graphs: {str(e)}")
            raise
    
    def _extract_centrality_features(self, df):
        """
        Extract centrality-based features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with centrality features
        """
        try:
            result_df = df.copy()
            
            if self.graph is None:
                logger.warning("Graph not built. Skipping centrality features.")
                return result_df
            
            # Calculate degree centrality
            in_degree_centrality = nx.in_degree_centrality(self.graph)
            out_degree_centrality = nx.out_degree_centrality(self.graph)
            degree_centrality = nx.degree_centrality(self.graph.to_undirected())
            
            # Map to dataframe
            if 'sender_id' in df.columns:
                result_df['sender_in_degree_centrality'] = df['sender_id'].map(in_degree_centrality).fillna(0)
                result_df['sender_out_degree_centrality'] = df['sender_id'].map(out_degree_centrality).fillna(0)
                result_df['sender_degree_centrality'] = df['sender_id'].map(degree_centrality).fillna(0)
            
            if 'receiver_id' in df.columns:
                result_df['receiver_in_degree_centrality'] = df['receiver_id'].map(in_degree_centrality).fillna(0)
                result_df['receiver_out_degree_centrality'] = df['receiver_id'].map(out_degree_centrality).fillna(0)
                result_df['receiver_degree_centrality'] = df['receiver_id'].map(degree_centrality).fillna(0)
            
            # Calculate betweenness centrality (sample for large graphs)
            if len(self.graph.nodes()) <= 1000:
                betweenness_centrality = nx.betweenness_centrality(self.graph)
            else:
                # Sample nodes for betweenness calculation
                sample_nodes = list(self.graph.nodes())[:1000]
                betweenness_centrality = nx.betweenness_centrality(self.graph, k=sample_nodes)
            
            # Map to dataframe
            if 'sender_id' in df.columns:
                result_df['sender_betweenness_centrality'] = df['sender_id'].map(betweenness_centrality).fillna(0)
            
            if 'receiver_id' in df.columns:
                result_df['receiver_betweenness_centrality'] = df['receiver_id'].map(betweenness_centrality).fillna(0)
            
            # Calculate closeness centrality (sample for large graphs)
            if len(self.graph.nodes()) <= 1000:
                closeness_centrality = nx.closeness_centrality(self.graph)
            else:
                # Use approximate closeness for large graphs
                closeness_centrality = nx.closeness_centrality(self.graph, distance='weight')
            
            # Map to dataframe
            if 'sender_id' in df.columns:
                result_df['sender_closeness_centrality'] = df['sender_id'].map(closeness_centrality).fillna(0)
            
            if 'receiver_id' in df.columns:
                result_df['receiver_closeness_centrality'] = df['receiver_id'].map(closeness_centrality).fillna(0)
            
            # Calculate eigenvector centrality (sample for large graphs)
            if len(self.graph.nodes()) <= 1000:
                try:
                    eigenvector_centrality = nx.eigenvector_centrality(self.graph, max_iter=1000)
                except:
                    eigenvector_centrality = {}
            else:
                eigenvector_centrality = {}
            
            # Map to dataframe
            if 'sender_id' in df.columns:
                result_df['sender_eigenvector_centrality'] = df['sender_id'].map(eigenvector_centrality).fillna(0)
            
            if 'receiver_id' in df.columns:
                result_df['receiver_eigenvector_centrality'] = df['receiver_id'].map(eigenvector_centrality).fillna(0)
            
            # Calculate PageRank
            pagerank = nx.pagerank(self.graph)
            
            # Map to dataframe
            if 'sender_id' in df.columns:
                result_df['sender_pagerank'] = df['sender_id'].map(pagerank).fillna(0)
            
            if 'receiver_id' in df.columns:
                result_df['receiver_pagerank'] = df['receiver_id'].map(pagerank).fillna(0)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting centrality features: {str(e)}")
            return df
    
    def _extract_clustering_features(self, df):
        """
        Extract clustering-based features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with clustering features
        """
        try:
            result_df = df.copy()
            
            if self.graph is None:
                logger.warning("Graph not built. Skipping clustering features.")
                return result_df
            
            # Calculate clustering coefficient for transaction graph
            clustering_coeff = nx.clustering(self.graph.to_undirected())
            
            # Map to dataframe
            if 'sender_id' in df.columns:
                result_df['sender_clustering_coefficient'] = df['sender_id'].map(clustering_coeff).fillna(0)
            
            if 'receiver_id' in df.columns:
                result_df['receiver_clustering_coefficient'] = df['receiver_id'].map(clustering_coeff).fillna(0)
            
            # Calculate clustering coefficient for sender graph
            if self.sender_graph is not None and len(self.sender_graph.nodes()) > 0:
                sender_clustering_coeff = nx.clustering(self.sender_graph)
                
                # Map to dataframe
                if 'sender_id' in df.columns:
                    result_df['sender_graph_clustering_coefficient'] = df['sender_id'].map(sender_clustering_coeff).fillna(0)
            
            # Calculate clustering coefficient for receiver graph
            if self.receiver_graph is not None and len(self.receiver_graph.nodes()) > 0:
                receiver_clustering_coeff = nx.clustering(self.receiver_graph)
                
                # Map to dataframe
                if 'receiver_id' in df.columns:
                    result_df['receiver_graph_clustering_coefficient'] = df['receiver_id'].map(receiver_clustering_coeff).fillna(0)
            
            # Calculate average neighbor degree
            avg_neighbor_degree = nx.average_neighbor_degree(self.graph.to_undirected())
            
            # Map to dataframe
            if 'sender_id' in df.columns:
                result_df['sender_avg_neighbor_degree'] = df['sender_id'].map(avg_neighbor_degree).fillna(0)
            
            if 'receiver_id' in df.columns:
                result_df['receiver_avg_neighbor_degree'] = df['receiver_id'].map(avg_neighbor_degree).fillna(0)
            
            # Calculate square clustering
            square_clustering = nx.square_clustering(self.graph.to_undirected())
            
            # Map to dataframe
            if 'sender_id' in df.columns:
                result_df['sender_square_clustering'] = df['sender_id'].map(square_clustering).fillna(0)
            
            if 'receiver_id' in df.columns:
                result_df['receiver_square_clustering'] = df['receiver_id'].map(square_clustering).fillna(0)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting clustering features: {str(e)}")
            return df
    
    def _extract_community_features(self, df):
        """
        Extract community-based features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with community features
        """
        try:
            result_df = df.copy()
            
            if self.graph is None:
                logger.warning("Graph not built. Skipping community features.")
                return result_df
            
            # Detect communities using Louvain algorithm
            undirected_graph = self.graph.to_undirected()
            communities = community_louvain.best_partition(undirected_graph)
            
            # Map to dataframe
            if 'sender_id' in df.columns:
                result_df['sender_community'] = df['sender_id'].map(communities).fillna(-1)
            
            if 'receiver_id' in df.columns:
                result_df['receiver_community'] = df['receiver_id'].map(communities).fillna(-1)
            
            # Calculate community size for each node
            community_sizes = Counter(communities.values())
            node_community_sizes = {node: community_sizes[comm] for node, comm in communities.items()}
            
            # Map to dataframe
            if 'sender_id' in df.columns:
                result_df['sender_community_size'] = df['sender_id'].map(node_community_sizes).fillna(0)
            
            if 'receiver_id' in df.columns:
                result_df['receiver_community_size'] = df['receiver_id'].map(node_community_sizes).fillna(0)
            
            # Calculate community degree centrality
            community_degree_centrality = {}
            for comm, nodes in community_sizes.items():
                comm_nodes = [node for node, c in communities.items() if c == comm]
                subgraph = undirected_graph.subgraph(comm_nodes)
                if len(subgraph.nodes()) > 0:
                    comm_centrality = nx.degree_centrality(subgraph)
                    for node in comm_nodes:
                        community_degree_centrality[node] = comm_centrality.get(node, 0)
            
            # Map to dataframe
            if 'sender_id' in df.columns:
                result_df['sender_community_degree_centrality'] = df['sender_id'].map(community_degree_centrality).fillna(0)
            
            if 'receiver_id' in df.columns:
                result_df['receiver_community_degree_centrality'] = df['receiver_id'].map(community_degree_centrality).fillna(0)
            
            # Check if sender and receiver are in the same community
            if 'sender_id' in df.columns and 'receiver_id' in df.columns:
                same_community = []
                for _, row in df.iterrows():
                    sender = row['sender_id']
                    receiver = row['receiver_id']
                    
                    if sender in communities and receiver in communities:
                        same_community.append(int(communities[sender] == communities[receiver]))
                    else:
                        same_community.append(0)
                
                result_df['same_community'] = same_community
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting community features: {str(e)}")
            return df
    
    def _extract_path_features(self, df):
        """
        Extract path-based features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with path features
        """
        try:
            result_df = df.copy()
            
            if self.graph is None:
                logger.warning("Graph not built. Skipping path features.")
                return result_df
            
            # Calculate shortest path lengths (sample for large graphs)
            if len(self.graph.nodes()) <= 500:
                # For small graphs, calculate all pairs shortest paths
                shortest_paths = dict(nx.all_pairs_shortest_path_length(self.graph))
                
                # Map to dataframe
                if 'sender_id' in df.columns and 'receiver_id' in df.columns:
                    path_lengths = []
                    for _, row in df.iterrows():
                        sender = row['sender_id']
                        receiver = row['receiver_id']
                        
                        if sender in shortest_paths and receiver in shortest_paths[sender]:
                            path_lengths.append(shortest_paths[sender][receiver])
                        else:
                            path_lengths.append(float('inf'))
                    
                    result_df['shortest_path_length'] = path_lengths
            else:
                # For large graphs, sample or use approximation
                if 'sender_id' in df.columns and 'receiver_id' in df.columns:
                    # Use BFS for a limited depth
                    path_lengths = []
                    for _, row in df.iterrows():
                        sender = row['sender_id']
                        receiver = row['receiver_id']
                        
                        try:
                            path_length = nx.shortest_path_length(self.graph, sender, receiver)
                            path_lengths.append(path_length)
                        except:
                            path_lengths.append(float('inf'))
                    
                    result_df['shortest_path_length'] = path_lengths
            
            # Calculate number of common neighbors
            if 'sender_id' in df.columns and 'receiver_id' in df.columns:
                common_neighbors = []
                for _, row in df.iterrows():
                    sender = row['sender_id']
                    receiver = row['receiver_id']
                    
                    try:
                        sender_neighbors = set(self.graph.predecessors(sender)) | set(self.graph.successors(sender))
                        receiver_neighbors = set(self.graph.predecessors(receiver)) | set(self.graph.successors(receiver))
                        common_neighbors.append(len(sender_neighbors & receiver_neighbors))
                    except:
                        common_neighbors.append(0)
                
                result_df['common_neighbors_count'] = common_neighbors
            
            # Calculate Jaccard coefficient
            if 'sender_id' in df.columns and 'receiver_id' in df.columns:
                jaccard_coeffs = []
                for _, row in df.iterrows():
                    sender = row['sender_id']
                    receiver = row['receiver_id']
                    
                    try:
                        sender_neighbors = set(self.graph.predecessors(sender)) | set(self.graph.successors(sender))
                        receiver_neighbors = set(self.graph.predecessors(receiver)) | set(self.graph.successors(receiver))
                        
                        union = sender_neighbors | receiver_neighbors
                        intersection = sender_neighbors & receiver_neighbors
                        
                        if len(union) > 0:
                            jaccard = len(intersection) / len(union)
                        else:
                            jaccard = 0
                        
                        jaccard_coeffs.append(jaccard)
                    except:
                        jaccard_coeffs.append(0)
                
                result_df['jaccard_coefficient'] = jaccard_coeffs
            
            # Calculate Adamic-Adar index
            if 'sender_id' in df.columns and 'receiver_id' in df.columns:
                adamic_adar = []
                for _, row in df.iterrows():
                    sender = row['sender_id']
                    receiver = row['receiver_id']
                    
                    try:
                        sender_neighbors = set(self.graph.predecessors(sender)) | set(self.graph.successors(sender))
                        receiver_neighbors = set(self.graph.predecessors(receiver)) | set(self.graph.successors(receiver))
                        common = sender_neighbors & receiver_neighbors
                        
                        # Calculate sum of 1/log(degree) for common neighbors
                        aa_index = 0
                        for node in common:
                            degree = self.graph.degree(node)
                            if degree > 1:
                                aa_index += 1 / np.log(degree)
                        
                        adamic_adar.append(aa_index)
                    except:
                        adamic_adar.append(0)
                
                result_df['adamic_adar_index'] = adamic_adar
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting path features: {str(e)}")
            return df
    
    def _extract_subgraph_features(self, df):
        """
        Extract subgraph-based features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with subgraph features
        """
        try:
            result_df = df.copy()
            
            if self.graph is None:
                logger.warning("Graph not built. Skipping subgraph features.")
                return result_df
            
            # Calculate ego network features for senders
            if 'sender_id' in df.columns:
                sender_ego_sizes = []
                sender_ego_densities = []
                
                for sender in df['sender_id']:
                    try:
                        # Get ego network
                        ego_graph = nx.ego_graph(self.graph.to_undirected(), sender, radius=1)
                        
                        # Calculate size and density
                        ego_size = len(ego_graph.nodes())
                        ego_density = nx.density(ego_graph)
                        
                        sender_ego_sizes.append(ego_size)
                        sender_ego_densities.append(ego_density)
                    except:
                        sender_ego_sizes.append(0)
                        sender_ego_densities.append(0)
                
                result_df['sender_ego_network_size'] = sender_ego_sizes
                result_df['sender_ego_network_density'] = sender_ego_densities
            
            # Calculate ego network features for receivers
            if 'receiver_id' in df.columns:
                receiver_ego_sizes = []
                receiver_ego_densities = []
                
                for receiver in df['receiver_id']:
                    try:
                        # Get ego network
                        ego_graph = nx.ego_graph(self.graph.to_undirected(), receiver, radius=1)
                        
                        # Calculate size and density
                        ego_size = len(ego_graph.nodes())
                        ego_density = nx.density(ego_graph)
                        
                        receiver_ego_sizes.append(ego_size)
                        receiver_ego_densities.append(ego_density)
                    except:
                        receiver_ego_sizes.append(0)
                        receiver_ego_densities.append(0)
                
                result_df['receiver_ego_network_size'] = receiver_ego_sizes
                result_df['receiver_ego_network_density'] = receiver_ego_densities
            
            # Calculate bipartite projection features
            if self.bipartite_graph is not None:
                # Get sender and receiver sets
                senders = {n for n, d in self.bipartite_graph.nodes(data=True) if d['bipartite'] == 0}
                receivers = {n for n, d in self.bipartite_graph.nodes(data=True) if d['bipartite'] == 1}
                
                # Project to sender graph
                sender_projection = nx.bipartite.projected_graph(self.bipartite_graph, senders)
                
                # Calculate features for sender projection
                if 'sender_id' in df.columns:
                    sender_proj_degrees = []
                    for sender in df['sender_id']:
                        try:
                            degree = sender_projection.degree(sender)
                            sender_proj_degrees.append(degree)
                        except:
                            sender_proj_degrees.append(0)
                    
                    result_df['sender_projection_degree'] = sender_proj_degrees
                
                # Project to receiver graph
                receiver_projection = nx.bipartite.projected_graph(self.bipartite_graph, receivers)
                
                # Calculate features for receiver projection
                if 'receiver_id' in df.columns:
                    receiver_proj_degrees = []
                    for receiver in df['receiver_id']:
                        try:
                            degree = receiver_projection.degree(receiver)
                            receiver_proj_degrees.append(degree)
                        except:
                            receiver_proj_degrees.append(0)
                    
                    result_df['receiver_projection_degree'] = receiver_proj_degrees
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting subgraph features: {str(e)}")
            return df
    
    def _extract_temporal_graph_features(self, df):
        """
        Extract temporal graph features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: DataFrame with temporal graph features
        """
        try:
            result_df = df.copy()
            
            if self.graph is None or 'timestamp' not in df.columns:
                logger.warning("Graph not built or timestamp not available. Skipping temporal graph features.")
                return result_df
            
            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df_sorted = df.sort_values('timestamp')
            
            # Calculate time window features
            time_windows = ['1H', '6H', '24H', '7D']
            
            for window in time_windows:
                # Calculate sender activity in time window
                if 'sender_id' in df.columns:
                    sender_activity = []
                    
                    for _, row in df_sorted.iterrows():
                        sender = row['sender_id']
                        timestamp = row['timestamp']
                        
                        # Get transactions in time window before current transaction
                        window_start = timestamp - pd.Timedelta(window)
                        window_end = timestamp
                        
                        window_transactions = df_sorted[
                            (df_sorted['timestamp'] >= window_start) &
                            (df_sorted['timestamp'] < window_end) &
                            (df_sorted['sender_id'] == sender)
                        ]
                        
                        # Calculate activity metrics
                        activity_count = len(window_transactions)
                        activity_amount = window_transactions['amount'].sum() if 'amount' in window_transactions.columns else 0
                        
                        sender_activity.append({
                            f'sender_activity_count_{window}': activity_count,
                            f'sender_activity_amount_{window}': activity_amount
                        })
                    
                    # Add to result dataframe
                    activity_df = pd.DataFrame(sender_activity)
                    for col in activity_df.columns:
                        result_df[col] = activity_df[col].values
                
                # Calculate receiver activity in time window
                if 'receiver_id' in df.columns:
                    receiver_activity = []
                    
                    for _, row in df_sorted.iterrows():
                        receiver = row['receiver_id']
                        timestamp = row['timestamp']
                        
                        # Get transactions in time window before current transaction
                        window_start = timestamp - pd.Timedelta(window)
                        window_end = timestamp
                        
                        window_transactions = df_sorted[
                            (df_sorted['timestamp'] >= window_start) &
                            (df_sorted['timestamp'] < window_end) &
                            (df_sorted['receiver_id'] == receiver)
                        ]
                        
                        # Calculate activity metrics
                        activity_count = len(window_transactions)
                        activity_amount = window_transactions['amount'].sum() if 'amount' in window_transactions.columns else 0
                        
                        receiver_activity.append({
                            f'receiver_activity_count_{window}': activity_count,
                            f'receiver_activity_amount_{window}': activity_amount
                        })
                    
                    # Add to result dataframe
                    activity_df = pd.DataFrame(receiver_activity)
                    for col in activity_df.columns:
                        result_df[col] = activity_df[col].values
            
            # Calculate time since last transaction for sender
            if 'sender_id' in df.columns:
                time_since_last = []
                
                for _, row in df_sorted.iterrows():
                    sender = row['sender_id']
                    timestamp = row['timestamp']
                    
                    # Get previous transaction from same sender
                    prev_transactions = df_sorted[
                        (df_sorted['timestamp'] < timestamp) &
                        (df_sorted['sender_id'] == sender)
                    ]
                    
                    if len(prev_transactions) > 0:
                        last_timestamp = prev_transactions['timestamp'].max()
                        time_diff = (timestamp - last_timestamp).total_seconds()
                        time_since_last.append(time_diff)
                    else:
                        time_since_last.append(float('inf'))
                
                result_df['sender_time_since_last_transaction'] = time_since_last
            
            # Calculate time since last transaction for receiver
            if 'receiver_id' in df.columns:
                time_since_last = []
                
                for _, row in df_sorted.iterrows():
                    receiver = row['receiver_id']
                    timestamp = row['timestamp']
                    
                    # Get previous transaction to same receiver
                    prev_transactions = df_sorted[
                        (df_sorted['timestamp'] < timestamp) &
                        (df_sorted['receiver_id'] == receiver)
                    ]
                    
                    if len(prev_transactions) > 0:
                        last_timestamp = prev_transactions['timestamp'].max()
                        time_diff = (timestamp - last_timestamp).total_seconds()
                        time_since_last.append(time_diff)
                    else:
                        time_since_last.append(float('inf'))
                
                result_df['receiver_time_since_last_transaction'] = time_since_last
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting temporal graph features: {str(e)}")
            return df
    
    def fit_transform(self, df):
        """
        Fit the feature extractor and transform the data
        
        Args:
            df (DataFrame): Input data
            
        Returns:
            DataFrame: Transformed data with features
        """
        # Extract features
        result_df = self.extract_features(df)
        
        # Get feature columns
        feature_cols = [col for col in result_df.columns if col not in df.columns]
        
        if len(feature_cols) > 0:
            # Fit scaler
            self.scaler.fit(result_df[feature_cols])
            
            self.fitted = True
        
        return result_df
    
    def transform(self, df):
        """
        Transform new data using fitted feature extractor
        
        Args:
            df (DataFrame): Input data
            
        Returns:
            DataFrame: Transformed data with features
        """
        if not self.fitted:
            raise ValueError("Feature extractor not fitted. Call fit_transform first.")
        
        # Extract features
        result_df = self.extract_features(df)
        
        # Get feature columns
        feature_cols = [col for col in result_df.columns if col not in df.columns]
        
        if len(feature_cols) > 0:
            # Transform features using fitted scaler
            result_df[feature_cols] = self.scaler.transform(result_df[feature_cols])
        
        return result_df