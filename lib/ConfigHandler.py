
import pandas as pd
import xml.etree.ElementTree as et 


class ConfigHandler():
    def __init__(self):
        self.xml_file = "./Utils/ConfigFile2018.xml"
        self.DF_COL = [ 'channel', 'start_run',
                        'end_run', 'detector', 'name', 
                        'mapping_row', 'mapping_column',
                        'delay', 'offset', 'HV',
                        'is_on', 'Vop']
        #convert the data type to desired type
        self.convert_dict = convert_dict = {'start_run': int, 
                                            'end_run': int,
                                            'detector': str,
                                            'name': str,
                                            'mapping_row':int,
                                            'mapping_column':int,
                                            'delay': int,
                                            'offset': int,
                                            'HV': float,
                                            'is_on':bool,
                                            'Vop':float
                                           } 
        self.parse_XML()
        #self.out_df = ''

    def parse_XML(self): 
        """Parse the input XML file and store the result in a pandas 
        DataFrame with the given columns. 

        The first element of df_cols is supposed to be the identifier 
        variable, which is an attribute of each node element in the 
        XML data; other features will be parsed from the text content 
        of each sub-element. 
        """
        df_cols = self.DF_COL
        xtree = et.parse(self.xml_file)
        xroot = xtree.getroot()
        rows = []

        for node in xroot: 
            res = []
            res.append(node.attrib.get(df_cols[0]))
            for el in df_cols[1:]: 
                if node is not None and node.find(el) is not None:
                    res.append(node.find(el).text)
                else: 
                    res.append(None)
            rows.append({df_cols[i]: res[i] 
                         for i, _ in enumerate(df_cols)})

        out_df = pd.DataFrame(rows, columns=df_cols)
        out_df['is_on'] = out_df['is_on'].apply(lambda x: True if x == 'true' else False)
        out_df = out_df.astype(self.convert_dict) 
        out_df = out_df.drop(['channel'], axis=1)
        self.out_df = out_df
    
    def get_run_info(self, run, detector):
        config = self.out_df[self.out_df['detector'] == detector]
        config = config[config['start_run'] <= run]
        config = config[config['end_run'] >= run]
        return config