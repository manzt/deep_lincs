class CsvLoader:
    def __init__(self, name=None, lookup_key=None, replace_header=None, **kwargs):
        self.name = name
        self.lookup_key=lookup_key
        self.replace_header=replace_header
        self.read_csv = partial(pd.read_csv, **kwargs)
    
    @classmethod
    def from_node(cls, node, data_dir):
        path = os.path.join(data_dir, node.pop("file"))
        return cls(filepath_or_buffer=path, **node)
        
    def read(self, **kwargs):
        df = self.read_csv(**kwargs)
        return df
    
class CsvMetaDataLoader:
    def __init__(self, main_loader, lookup_loaders=[]):
        self.main_loader = main_loader
        self.lookup_loaders = lookup_loaders
    
    def read(self, idx=None, ids=None):
        df = self.main_loader.read()
        df = self.subset_df(df, idx, ids)
        for lookup_loader in self.lookup_loaders:
            cache_index = df.index.name
            df = df.reset_index().merge(
                lookup_loader.read(),
                left_on=lookup_loader.lookup_key,
                right_on=lookup_loader.lookup_key,
                how="left"
            ).set_index(cache_index)
        return df
    
    @classmethod
    def from_node(cls, node, data_dir):
        main_csv = CsvLoader.from_node(node["main"], data_dir)
        lookup_csvs = [
            CsvLoader.from_node(lookup_node, data_dir) for lookup_node in node["lookup"]
        ] if "lookup" in node else []
        return cls(main_csv, lookup_csvs)
    
    @staticmethod
    def subset_df(df, idx, ids):
        if idx and ids:
            raise ValueError("Must use index or id values. Cannot use both.")
        if idx:
            return df.iloc[idx,:]
        elif ids:
            mask = df.index.isin(ids)
            return df.loc[mask,:]
        else:
            return df
    
    def __repr__(self):
        return f"< CsvMetaDataLoader: main: {self.main_loader.name} >"