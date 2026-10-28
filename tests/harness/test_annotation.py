"""Public annotation regression: grouping identity must survive pandas 2/3 apply."""
from pathlib import Path
import pandas as pd
import pytest
from glossapi import Corpus
from glossapi.gloss_section_classifier import GlossSectionClassifier


@pytest.fixture
def sections():
    # Interleaved, unsorted document groups; no heuristic index override. Distinct labels exercise full-annotation boundaries.
    rows=[]
    for name in ('alpha','beta'):
        for i,label in ((4,'άλλο'),(2,'π'),(1,'άλλο'),(5,'άλλο'),(3,'β')):
            rows.append(dict(filename=name,id=i,header='Header',section='Body text',
                section_length=10,section_propo=10,has_table=False,has_list=False,
                expected_prediction=label))
    return pd.DataFrame(rows)


def test_annotation_preserves_filename_columns_and_values(tmp_path,monkeypatch,sections):
    repo=Path(__file__).resolve().parents[2]
    corpus=Corpus(tmp_path/'input',tmp_path/'out',
        section_classifier_model_path=repo/'src/glossapi/models/section_classifier.joblib',
        extraction_model_path=repo/'src/glossapi/models/kmeans_weights.joblib')
    corpus.sections_parquet.parent.mkdir(parents=True,exist_ok=True)
    sections.to_parquet(corpus.sections_parquet,index=False)
    # Fix only the model's decisions, exercising real Dask parquet partitions,
    # grouping/postprocessing and the public annotate implementation.
    class Model:
        def predict(self,x):
            return [sections.loc[(sections.header==r.header)&(sections.section==r.section),'expected_prediction'].iloc[0] for r in x.itertuples()]
    # Distinct header/section tokens let the deterministic model identify rows.
    sections['header']=sections.apply(lambda r:f'{r.filename}-{r.id}',axis=1)
    sections.to_parquet(corpus.sections_parquet,index=False)
    monkeypatch.setattr(GlossSectionClassifier,'load_model',lambda self,*a,**kw:Model())
    corpus.annotate()
    classified=pd.read_parquet(corpus.classified_parquet)
    annotated=pd.read_parquet(corpus.fully_annotated_parquet)
    assert classified.columns.tolist()==sections.columns.tolist()+['predicted_section']
    assert annotated.columns.tolist()==classified.columns.tolist()+['processing_stage']
    assert annotated.filename.tolist()==['alpha']*5+['beta']*5
    assert annotated.id.tolist()==[1,2,3,4,5]*2
    assert annotated.predicted_section.tolist()==['ε.σ.','π','β','α','α']*2
    assert annotated.processing_stage.tolist()==['section,annotate']*10
    assert annotated.header.tolist()==[f'{name}-{i}' for name in ('alpha','beta') for i in range(1,6)]
    assert annotated.section.tolist()==['Body text']*10
