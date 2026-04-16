from confluent_kafka import Consumer

consumer = Consumer({
    'bootstrap.servers': "localhost:9092",
    'group.id': 'test-group',
    'auto.offset.reset': 'earliest'
})

print("Connected to Kafka!")