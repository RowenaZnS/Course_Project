import fs from 'fs/promises';
import client from './dbclient.js';
async function fetch_orders(){
    try {
        const users = client.db('lab5db').collection('users');
        const query = {};
        const all_list = await users.find(query).toArray();
       // console.log(all_list);
        if(all_list==null)return false;
        else return all_list;
    } catch (error) {
        console.log("Unable to fetch from database!");
        return null;
    } 
}
async function fetch_movie() {
    try {
        const users = client.db('lab5db').collection('movielist');
        const query = {};
        const all_list = await users.find(query).toArray();
       // console.log(all_list);
        if(all_list==null)return false;
        else return all_list;
    } catch (error) {
        console.log("Unable to fetch from database!");
        return null;
    } 
}
async function 
update_movie(image,name,price, date, venue, introduction) {
    try {
        
        const users = client.db('lab5db').collection('movielist');
        const query = { name: name };
        const update = {
            $set: {
                image:image,
                name:name,
                price:price,
                date:date,
                venue:venue,
                introduction: introduction,
            }
        };
        const options = { upsert: true };
        const result = await users.updateOne(query, update, options);
        if (result.upsertedCount > 0) {
            console.log("Added 1 user");
        } else {
            console.log("Added 0 user");
        }
        return true;
    }catch(err){
        console.log("Unable to update the database!");
    }
}
export {fetch_movie,update_movie,fetch_orders};