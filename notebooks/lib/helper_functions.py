import pandas as pds
from typing import Optional, List, Any
from rdflib import URIRef, BNode, Literal, Graph, Namespace, RDF, RDFS, OWL
from rdflib.plugins.sparql.processor import SPARQLResult

def init_graph(ontology:str =None, format='ttl') -> Graph:
    """attaches custom functions an rdflib Graph and returns a new graph"""
    
    # attach helper functions
    Graph.add_spo = add_spo
    Graph.add_spv = add_spv
    Graph.sparql_query_to_df = sparql_query_to_df
    
    # intstantiate Graph
    graph = Graph()
    
    # parse ontology
    if ontology:
        return graph.parse(ontology, format=format)
    else:
        return graph
    
    
def add_spo(self, subj: Any, predicate: Any, obj: Any) -> Graph:
    """shortcut for adding subject, predicate, object URIRefs to graph"""
    
    self.add((URIRef(subj), URIRef(predicate), URIRef(obj)))


def add_spv(self, subj: Any, predicate: Any, val: Any) -> Graph:
    """shortcut for adding subject, predicate URIRefs and literal value to graph"""
    
    self.add((URIRef(subj), URIRef(predicate), Literal(val)))


def sparql_results_to_df(results: SPARQLResult, graph=Optional[Graph]) -> pds.DataFrame:
    """converts sparql results into a dataframe"""
    
    def set_value(x):
        if x is None:
            return None
        elif graph is not None:
            for n in graph.namespaces():
                # each is a tuple of form (<prefix>, URIRef(...))
                # e.g., ('dc', rdflib.term.URIRef('http://purl.org/dc/elements/1.1/'))
                if str(x).startswith(str(n[1])):
                    # replace uri with prefix
                    return str(x).replace(n[1], n[0])
                
            # if it makes it here, no replacements occurred
            return x.toPython()
        else:
            return x.toPython()

    return \
        pds.DataFrame(
            data=([set_value(x) for x in row] for row in results),
            columns=[str(x) for x in results.vars]
        )


def sparql_query_to_df(query: str, graph: Graph, use_ns=True) -> pds.DataFrame:
    """queries the graph and returns the results as a dataframe"""
    
    results = graph.query(query)
    if use_ns:
        return sparql_results_to_df(results, graph)
    else:
        return sparql_results_to_df(results, None)


def add_table_metadata_to_graph(table: pds.DataFrame, 
                                table_name: str, 
                                graph: Graph, 
                                table_ns: Namespace, 
                                field_ns: Namespace, 
                                property_ns: Namespace) -> Graph:
    """adds instances of tables and fields to graph"""
    
    # add table instance to graph
    table_uri = table_ns[f'/{table_name}']
    graph = add_spo(graph, table_uri, RDF.type, table_ns)
    graph = add_spv(graph, table_uri, RDFS.label, table_name)
        
    # add each of the tables fields to graph as instances and subclass of fields
    for field_name in table.columns:
        field_name = f'{table_name}.{field_name}' # prepend table name to field name
        uri = URIRef(field_ns[f'/{field_name}'])
        graph = add_spo(graph, uri, RDF.type, field_ns)
        grpah = add_spo(graph, uri, property_ns.member_of, table_uri)
        graph = add_spv(graph, uri, RDFS.label, field_name)
        
        # *pun* the field as an owl class
        # note: field_ns[:-1] removes the last "/" from the uri
        graph = add_spo(graph, uri, RDF.type, OWL.Class)
        graph = add_spo(graph, uri, RDFS.subClassOf, field_ns[:-1])
    
    return graph


def add_enums_to_graph(enums: List, 
                       table_name: str, 
                       field_name: str, 
                       graph: Graph, 
                       enum_ns: Namespace, 
                       base_ns: Namespace) -> Graph:
    """adds instances of enumerated values to graph"""
    
    field_name = f'{table_name}.{field_name}' # prepend table name to field name
    for enum in enums:
        # build uri
        uri = URIRef(enum_ns[f'/{field_name}#{enum}'])

        # add instances
        # Note: the literal value is added to the graph as well
        graph = add_spo(graph, uri, RDF.type, enum_ns)
        graph = add_spv(graph, uri, base_ns.has_value, enum)
        graph = add_spv(graph, uri, RDFS.label, f'{field_name} {enum}')

        # enums constrain values in fields, so add this informaton the graph
        field = base_ns[f'field/{field_name}']
        graph = add_spo(graph, uri, base_ns.defines_values_in, field)
        
    return graph


def df_to_sql(df: pds.DataFrame) -> str:
    """creates an sql query based on values in a dataframe (with specific columns)"""
    # N.B.: This need to be refactored!!!
    
    # dict to hold info about the table and class the field represents
    # note: check for enum_value column
    if 'enum_value' in df.columns:
        columns = ['field_name', 'table_name', 'cls_names', 'enum_value']
    else:
        columns = ['field_name', 'table_name', 'cls_names']
        
    field_info_dict = \
        df[columns].drop_duplicates().set_index('field_name').to_dict(orient='index')
    
    
    # gather all table and field names in dataframe
    all_tables = set(field_df['table_name'])
    all_fields = set(field_df['field_name'])
    
    # dict to hold whether the table occurs in a 'from' clause or 'join' clause
    table_clause_dict = {'from': '', 'joins': []}
    for i, tbl in enumerate(all_tables):
        if i == 0:
            table_clause_dict['from'] = tbl
        else:
            table_clause_dict['joins'].append(tbl)
    
    # dict to hold just those tables that occur in 'join' clause
    # this dict just help build the sql query
    table_join_dict = \
        {tbl:{'joins': []} for tbl in all_tables if tbl != table_clause_dict['from']}
    
    # iterate over each class in the dataframe
    # b/c each field is part of the same group we know the reprenst the same type of thing
    # so, we can relate the fields in the join clause
    for group_name, group_df in df.groupby('cls_names'):
        # all fields in the same group represent the same type of thing
        # find the pairwise combinations of how each field relates to one another
        field_pairs = \
            list(itertools.combinations(group_df['field_name'], 2))
        
        # now for each pair add it to the appropriate join table
        for left_field, right_field in field_pairs:
            # fetch table names of fields
            left_table = field_info_dict[left_field]['table_name']
            right_table = field_info_dict[right_field]['table_name']
            
            # add pair as a join condition (e.g. patients.patient_id = procedures.patient_id)
            if left_table != table_clause_dict['from']:
                table_join_dict[left_table]['joins'].append((left_field, right_field))
            elif right_table != table_clause_dict['from']:
                table_join_dict[right_table]['joins'].append((left_field, right_field))

    # finally, build sql query
    sql = ""
    
    # SELECT ...
    # form the select list for the sql
    select_fields = \
        [f"  {field} as [{field} ({value['cls_names']})]\n" 
        for field, value in field_info_dict.items()]
    
    select_fields = []
    for field, value in field_info_dict.items():
        # the "as [{field}" is needed to guarantee that the whole
        # field name (i.e., <table>.<field name>) is used in the results
        if 'cls_names' in value.keys():
            select_fields.append(f"  {field} as [{field} ({value['cls_names']})] \n")
        else:
            select_fields.append(f"  {field} as [{field}]\n)")
                    
    sql = sql + f"select \n{'  ,'.join(select_fields)} \n"
    
    # FROM ... JOIN ...
    # add tables that data is retrieved from (note: there must be a 'from' clause)
    if len(table_clause_dict['from']) > 0:
        sql = sql + f"from {table_clause_dict['from']} \n"
        
        #  collect joined tables
        for tbl in table_clause_dict['joins']:
            join_fields = table_join_dict[tbl]['joins']
            if len(join_fields) > 0:
                sql = sql + f"inner join {tbl} on \n"
                
                # put an '=' between each pair of fields
                field_pairs = [f"{' = '.join(field_pair)}\n" for field_pair in join_fields]
                sql = sql + f"  {'  and '.join(field_pairs)}"
    
    # WHERE ...
    # check if enum value used for filter
    if 'enum_value' in df.columns:
        # 
        print_where = True
        for idx, field_name, enum_value in df[['field_name', 'enum_value']].itertuples():
            if print_where and enum_value is not None and len(enum_value) > 0:
                sql = sql + f"where {field_name} = '{enum_value}' \n"
                print_where = False
            elif enum_value is not None and len(enum_value) > 0:
                sql = sql + f"and {field_name} = '{enum_value}' \n"
                
    # return sql query
    return sql


def add_dataframe_to_graph(graph: Graph, df: pds.DataFrame, table_name, field_ns: Namespace, base_ns: Namespace) -> Graph:
    """adds a simple rdf transformation of a dataframe into the graph"""
    
    for row in df.itertuples(index=None):
        # add row as blank node to table
        row_uri = BNode() 
        table_uri = base_ns[f'table/{table_name}']
        graph = add_spo(graph, row_uri, RDF.type, base_ns.row)
        graph = add_spo(graph, row_uri, base_ns.member_of, table_uri)
        
        # for each field in row
        for field_name, value in row._asdict().items():
            # add instance of field value using blank node
            field_value_uri = BNode()
            graph = add_spo(graph, field_value_uri, RDF.type, base_ns.field_value)
            graph = add_spv(graph, field_value_uri, base_ns.has_value, value)
                        
            # relate row to field value using *punned* field name
            field_uri = field_ns[f'/{table_name}.{field_name}']
            graph = add_spo(graph, field_uri, RDF.type, OWL.ObjectProperty)
            graph = add_spo(graph, row_uri, field_uri, field_value_uri)
            
            # relate field value to field and field to row via 'member of'
            field_instance_uri = BNode()
            graph = add_spo(graph, field_instance_uri, RDF.type, field_uri)
            graph = add_spo(graph, field_instance_uri, base_ns.member_of, row_uri)
            graph = add_spo(graph, field_value_uri, base_ns.member_of, field_instance_uri)

    return graph