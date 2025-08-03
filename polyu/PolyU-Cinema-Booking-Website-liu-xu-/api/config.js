import dotenv from 'dotenv';
dotenv.config();
console.log(process.env.MONGODB_URI);
if(process.env.MONGODB_URI==''){
    console.log("MONGODB_URI is not defined");
    process.exit(1);
}
export default {
    // Export variable here
    MONGODB_URI:process.env.MONGODB_URI
   }












