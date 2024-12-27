from preprocess.config.config import Conf
from preprocess.clinical import ClinicalData

clinical_data = ClinicalData(Conf.datasets['Clinical'])
clinical_data.preprocess_data()