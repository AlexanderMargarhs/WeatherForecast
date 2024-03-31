from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, request, jsonify

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///forecast.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Forecast(db.Model):
    formatted_date = db.Column(db.String, primary_key=True)
    temperature = db.Column(db.Float, nullable=False)
    apparent_temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    visibility = db.Column(db.Float, nullable=False)
    pressure = db.Column(db.Float, nullable=False)
    prediction = db.Column(db.String, nullable=False)

    def __repr__(self):
        return f'<Date {self.formatted_date}: Temp={self.temperature}, Apparent Temperature={self.apparent_temperature}, Humidity={self.humidity}, Visibility={self.visibility}, Pressure={self.pressure}, Prediction={self.prediction}>'


# Create the database tables before running the application
with app.app_context():
    db.create_all()


@app.route('/forecast', methods=['GET'])
def get_forecast():
    forecasts = Forecast.query.all()
    forecast_data = [{'formatted_date': forecast.formatted_date,
                      'temperature': forecast.temperature,
                      'apparent_temperature': forecast.apparent_temperature,
                      'humidity': forecast.humidity,
                      'visibility': forecast.visibility,
                      'pressure': forecast.pressure,
                      'prediction': forecast.prediction}
                     for forecast in forecasts]
    return jsonify(forecast_data)


@app.route('/forecast', methods=['POST'])
def post_forecast():
    data = request.get_json()
    formatted_date = datetime.strptime(data['formatted_date'], '%Y-%m-%d %H:%M:%S')
    new_forecast = Forecast(
        formatted_date=formatted_date,
        temperature=data['temperature'],  # Ensure temperature is provided
        apparent_temperature=data['apparent_temperature'],
        humidity=data['humidity'],
        visibility=data['visibility'],
        pressure=data['pressure'],
        prediction=data['prediction']
    )
    db.session.add(new_forecast)
    db.session.commit()
    return jsonify({'message': 'Forecast added successfully'}), 201


if __name__ == '__main__':
    app.run(debug=True)
