import flask_server
import server.learningAlgorithm
import tensorflow as tf

if __name__ == '__main__':
    # exec(open("server/learningAlgorithm.py").read())
    graph = tf.get_default_graph()
    model = server.learningAlgorithm.get_model()
    flask_server.run_flask_server('localhost', 5000, True, model, graph)
