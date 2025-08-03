/* eslint-disable no-unreachable */
import fs from 'fs/promises';
import client from './dbclient.js';
async function updateorder(firstName,lastName,email,address,region,district,paymentMethod,username,order_id,Purchase_Date,Event_id,Event_Name,Event_Date,seat_list,total_cost,tickets) {
    try {
        const users = client.db('lab5db').collection('users');
        const query = { username: username };
        const newOrder = {
            order_id: order_id,       // 新订单 ID
            firstName:firstName,
            lastName:lastName,
            email:email,
            address:address,
            region:region,
            district:district,
            paymentMethod:paymentMethod,
            Purchase_Date: Purchase_Date,
            Event_id: Event_id,
            Event_Name: Event_Name,
            Event_Date: Event_Date,
            seat_list: seat_list,
            total_cost: total_cost,
            tickets: tickets
        };
        const update = {
            $push: { order: newOrder }
        };
        const options = { upsert: true };
        const result = await users.updateOne(query, update, options);
        if (result.upsertedCount > 0) {
            console.log("Added 1 user");
        } else {
            console.log("Added 0 user");
        }
        return true;
    } catch (error) {
        console.log("Unable to fetch from database!");
        return null;
    } 
}
export {updateorder};