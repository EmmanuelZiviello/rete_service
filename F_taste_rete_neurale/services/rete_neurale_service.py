from F_taste_rete_neurale.ml_model.model import pre_process_data,predict


class ReteNeuraleService:
    
    @staticmethod
    def predizione(data_to_preprocess):
        data_pre_processed = pre_process_data(data_to_preprocess)

        prediction = predict(data_pre_processed)
        

        return {"outcome":prediction}, 200
        