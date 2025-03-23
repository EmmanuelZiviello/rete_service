from flask import jsonify, request
from flask_restx import Resource, ValidationError, fields
from flask_jwt_extended import get_jwt_identity
from F_taste_rete_neurale.namespaces import nutrizionista_ns
from F_taste_rete_neurale.services.rete_neurale_service import ReteNeuraleService
from F_taste_rete_neurale.utils.jwt_custom_decorators import nutrizionista_required


vector_model = nutrizionista_ns.model('vector for prediction model', {
    'glucose': fields.Float(required = True),
    'triglycerides': fields.Float(required = True),
    'hdl': fields.Float(required = True),
    'systolic': fields.Float(required = True),
    'diastolic': fields.Float(required = True),
    'gender': fields.Float(required = True),
    'age': fields.Float(required = True),
    'weight': fields.Float(required = True),
    'bmi': fields.Float(required = True)
}, strict = True)

class Predizione(Resource):

    @nutrizionista_ns.expect(vector_model)
    @nutrizionista_ns.doc('get prediction')
    @nutrizionista_required()
    def post(self):

        data_to_preprocess = [request.json['glucose'], request.json['triglycerides'], request.json['hdl'],
                          request.json['systolic'], request.json['diastolic'], request.json['gender'],
                          request.json['age'], request.json['weight'], request.json['bmi']]
        
        return ReteNeuraleService.predizione(data_to_preprocess)