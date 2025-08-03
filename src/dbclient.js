import { MongoClient, ServerApiVersion } from 'mongodb';
import config from './config.js';
const connect_uri = config.MONGODB_URI;
const client = new MongoClient(connect_uri, {
    connectTimeoutMS: 4000,
    serverSelectionTimeoutMS: 4000,
    serverApi: {
        version: ServerApiVersion.v1,
        strict: true,
        deprecationErrors: true,
    },
});
async function connect() {
    try {
        await client.connect();
        const db = client.db('lab5db');
        client.db("lab5db").command({ping: 1});
        console.log('Successfully connected to the database!');
    } catch (err) {
        console.error('Unable to establish connection to the database!',err);
        process.exit(1);
    }
}
connect().catch(console.dir);
client.db("lab5db").command({ping: 1});
export default client;
