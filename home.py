import os

import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bio
import dash_bio_utils.ngl_parser as ngl_parser
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd

from github import Github


from app import app

github = Github('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')
repository = github.get_organization('CapraLab').get_repo('cosmis_app')


# load data
pdb_path = 'pdbs/'
data_path = ''
# dataset_name = 'https://github.com/CapraLab/cosmis_app/blob/main/cosmis_dash.tsv'
cosmis_dataset = repository.get_content('cosmis_dash.tsv')
cosmis_df = pd.read_csv(
    # os.path.join(data_path, dataset_name),
    cosmis_dataset.decoded_content.decode(),
    sep='\t',
    header=0
)
hgnc_to_uniprot = {}
with open(os.path.join(data_path, 'hgnc_to_uniprot.tsv'), 'rt') as in_f:
    for l in in_f:
        x, y = l.strip().split('\t')
        hgnc_to_uniprot[x] = y

# uniprot_id to structure mapping
map_file = 'uniprot_to_struct.tsv'
uniprot_to_struct = pd.read_csv(
    os.path.join(data_path, map_file),
    sep='\t',
    header=0,
    index_col='uniprot_id'
)

# layout
home_layout = html.Div(
    [
        html.Div(
            [
                html.Span(
                    '''
                    The COntact Set MISsense (COSMIS) tolerance framework is a new method
                    to quantify the constraint on protein-coding genetic variation in 3D 
                    spatial neighborhoods. The 3D spatial neighboorhood of a residue is
                    defined by the set of residues that are in direct 3D interaction with
                    the residue of interest. The central hypothesis of COSMIS is that 
                    amino acid sites connected through direct 3D interactions collectively 
                    shape the level of constraint on each site. 
                    It leverages recent advances in computational structure prediction, large-scale 
                    sequencing data from gnomAD, and a mutation-spectrum-aware statistical model. 
                    The framework currently maps the landscape of 3D spatial constraint on 6.1 
                    amino acid sites covering >80% (16,533) of human proteins. As genetic
                    variation database and protein structure databases grow, we will continuously
                    update COSMIS.
                    '''
                    ),
            ],
            style={'padding': 20},
        ),
        html.Div(
            [
                html.H4(
                    'Search by UniProt ID or Gene Name',
                    style={'padding-left': 20 , 'margin-bottom':0, 'color': 'grey'}
                ),
                dbc.Row(
                        [
                        dbc.Col(
                            [
                                dbc.Input(
                                    type='search',
                                    placeholder='P51787 or KCNQ1',
                                    id='uniprot-id',
                                    bs_size='sm',
                                    style={'width':'70%'}
                                ),
                                html.Button(
                                    'Submit', id='search-button', n_clicks=0,
                                    style={'margin': '0px 10px 0px 10px'}
                                ),

                            ], style={'padding-left': 10, 'display': 'flex'}
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.Span('Slide to set a threshold:'),
                                        dcc.Slider(
                                            id='cosmis-slider',
                                            min=-8,
                                            max=8,
                                            value=8,
                                            step=0.25,
                                            marks={x: str(x) for x in range(-8, 9, 2)},
                                        ),
                                    ],
                                    style={
                                        'padding': 10
                                    },
                                ),
                            ],
                        ),
                        ],
                    no_gutters=True,
                    className='ml-auto mt-3 mt-md-0',
                    align='center',
                    style={'padding': '0px 10px 0px 10px'},
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Tabs(
                                    [
                                        dbc.Tab(label='COSMIS plot', tab_id='cosmis-plot-left'),
                                        dbc.Tab(label='3D structure view', tab_id='3d-view-left'),
                                        dbc.Tab(label='Tabular view', tab_id='cosmis-table-left'),
                                    ],
                                    id='tabs-left',
                                    active_tab='cosmis-plot-left',
                                    style={'padding': '0 10 0 10'},
                                ),
                                html.Div(
                                    id='tab-content-left', className='p-4'
                                ),
                            ], xs=12,sm=12,md=6,lg=6,xl=6),
                        dbc.Col(
                            [
                                dbc.Tabs(
                                    [
                                        dbc.Tab(label='COSMIS plot', tab_id='cosmis-plot-right'),
                                        dbc.Tab(label='3D structure view', tab_id='3d-view-right'),
                                        dbc.Tab(label='Tabular view', tab_id='cosmis-table-right'),
                                    ],
                                    id='tabs-right',
                                    active_tab='3d-view-right',
                                    style={'padding': '0 20 0 20'},
                                ),
                                html.Div(
                                    id='tab-content-right', className='p-4'
                                ),
                            ], xs=12,sm=12,md=6,lg=6,xl=6)
                    ],
                    style={'padding': '0px 20px 0px 20px'},
                ),
        ],
            style={
                'padding': '20px 0px 20px 0px',
                'border': '1px grey dashed',
                'border-radius': '8px',
                'margin': '0px 20px 0px 20px'
            },
        ),
        html.Div(
            [
                html.Hr(),
            ],
            style={
                'padding': '0px 20px 0px 20px',
            },
        ),
        html.Div([
            html.H4(
                'Batch Query by Variant IDs',
                style={'padding-left': 20 , 'margin-bottom': 0, 'color': 'grey'}
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Textarea(
                                id='input-variants',
                                placeholder='P51787 150 A B\nOne per line',
                                style={'width': '100%', 'height': 200},
                            ),
                        ],
                        style={'display': 'flex', 'padding': 10}
                    ),
                    dbc.Col(
                        [
                            dcc.Upload(
                                id='upload-data',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select Files')
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '200px',
                                    'lineHeight': '200px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'padding': '0px'
                                },
                                # Allow multiple files to be uploaded
                                multiple=True
                            ),
                            html.Div(id='output-data-upload'),
                        ], xs=12,sm=12,md=6,lg=6,xl=6),
                ],
                no_gutters=True,
                className='ml-auto mt-3 mt-md-0',
                align='center',
                style={'padding': '0px 10px 0px 10px'},
            ),
            html.Button(
                'Submit',
                id='variant-query-button',
                n_clicks=0,
                style={'margin': '0px 20px 0px 20px'},
            ),
            html.Div(
                id='variant-cosmis-table',
                style={'margin': '20px 20px 20px 20px'}
            ),
        ],
            style={
                'padding': '20px 0px 20px 0px',
                'border': '1px grey dashed',
                'border-radius': '8px',
                'margin': '0px 20px 0px 20px'
            },
        ),
        dcc.Store(id='store'),
    ],
)


@app.callback(
    Output('tab-content-left', 'children'),
    [
        Input('tabs-left', 'active_tab'),
        Input('store', 'data')
    ]
)
def render_tab_content_left(active_tab, data):
    if active_tab and data is not None:
        if active_tab == 'cosmis-plot-left':
            return dcc.Graph(figure=data['cosmis-plot'])
        elif active_tab == 'cosmis-table-left':
            return data['cosmis-table']
        elif active_tab == '3d-view-left':
            return data['3d-view']
    return 'No data retrieved'


@app.callback(
    Output('tab-content-right', 'children'),
    [
        Input('tabs-right', 'active_tab'),
        Input('store', 'data')
    ]
)
def render_tab_content_right(active_tab, data):
    if active_tab and data is not None:
        if active_tab == 'cosmis-plot-right':
            return dcc.Graph(figure=data['cosmis-plot'])
        elif active_tab == 'cosmis-table-right':
            return data['cosmis-table']
        elif active_tab == '3d-view-right':
            return data['3d-view']
    return 'No data retrieved'


def generate_table(dataframe, max_rows=10):
    return dash_table.DataTable(
        columns=[
            {'name': col, 'id': col} for col in [
                'uniprot_id', 'enst_id', 'uniprot_pos', 'uniprot_aa', 'cosmis', 'p_value'
            ]
        ],
        page_size=max_rows,
        data=dataframe.to_dict('records')
    )


# textarea callback
@app.callback(
    Output('variant-cosmis-table', 'children'),
    [
        Input('input-variants', 'value'),
        Input('variant-query-button', 'n_clicks'),
    ]
)
def retrieve_variant_cosmis(input_variants, n):
    if not n:
        return None

    indices = []
    for variant in input_variants.split('\n'):
        uniprot_id = variant.split()[0]
        uniprot_pos = variant.split()[1]
        indices.append(uniprot_id + '_' + uniprot_pos)

    # create new indices
    new_df= cosmis_df.astype({'uniprot_pos': str})
    new_df.set_index(
        new_df[['uniprot_id', 'uniprot_pos']].agg('_'.join, axis=1), inplace=True
    )

    # variant rows
    variant_df = new_df.loc[indices]

    # make a data table
    return generate_table(variant_df, max_rows=10)


@app.callback(
    Output('store', 'data'),
    [
        Input('uniprot-id', 'value'),
        Input('cosmis-slider', 'value'),
        Input('search-button', 'n_clicks')
    ]
)
def generate_graphs(given_uniprot_id, cosmis_cutoff, n):
    if not n:
        return None

    if given_uniprot_id in hgnc_to_uniprot.keys():
        uniprot_id = hgnc_to_uniprot[given_uniprot_id]
    else:
        uniprot_id = given_uniprot_id

    # extract the correct data
    filtered_df = cosmis_df[
        (cosmis_df['uniprot_id'] == uniprot_id) & (cosmis_df['cosmis'] <= cosmis_cutoff)
    ]

    # make a scatter plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered_df['uniprot_pos'],
            y=filtered_df['cosmis'],
            mode='markers',
            marker_size=7,
        )
    )
    for i in range(len(filtered_df)):
        fig.add_shape(
            type='line',
            x0=filtered_df['uniprot_pos'].iloc[i],
            x1=filtered_df['uniprot_pos'].iloc[i],
            y0=0,
            y1=filtered_df['cosmis'].iloc[i],
            line=dict(
                color='grey',
                width=1
            )
        )
    fig.update_layout(
        transition_duration=500,
        plot_bgcolor='white'
    )
    fig.update_xaxes(
        ticks='outside',
        linecolor='black',
        linewidth=1,
        title='Sequence position'
    )
    fig.update_yaxes(
        ticks='outside',
        linecolor='black',
        linewidth=1,
        range=[-8, 8],
        title='COSMIS score'
    )

    # create and NGL view
    atoms = ','.join([str(x) for x in filtered_df['uniprot_pos']])
    struct_id = uniprot_to_struct.loc[uniprot_id, 'struct_id']
    struct_source = uniprot_to_struct.loc[uniprot_id, 'struct_source']
    if struct_source == 'PDB':
        selected_atoms = struct_id + '.' + struct_id[4:] + ':1-5000@' + atoms
    elif struct_source == 'SWISS-MODEL':
        selected_atoms = struct_id + '.' + struct_id[-1] + ':1-5000@' + atoms
    else:
        selected_atoms = struct_id + '.A' + ':1-5000@' + atoms

    data_list = [
        ngl_parser.get_data(
            data_path=pdb_path,
            pdb_id=selected_atoms,
            color='blue',
            reset_view=True,
            local=True
        )
    ]
    molstyles_dict = {
        'representations': ['cartoon'],
        'chosenAtomsColor': 'red',
        'chosenAtomsRadius': 1,
        'molSpacingXaxis': 100
    }
    image_parameters = {
        'antialias': True,
        'transparent': True,
        'trim': True,
        'defaultFilename': 'tmp'
    }

    ngl_view = dash_bio.NglMoleculeViewer(
        data=data_list,
        molStyles=molstyles_dict,
        imageParameters=image_parameters
    )

    # make a table
    cosmis_table = generate_table(filtered_df, max_rows=10)

    return {'cosmis-plot': fig, '3d-view': ngl_view, 'cosmis-table': cosmis_table}
