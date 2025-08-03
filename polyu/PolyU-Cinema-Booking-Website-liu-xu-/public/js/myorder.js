$(document).ready(function () {
  // <!-- XU Zhuoning liu xinyan 21101256d;22097739d -->
    $.ajax({
      url: '/auth/myorder',
      type: 'GET',
      success: function (data) {
        if (data.status === 'success') {
          $('#username').text(data.user.username);
          $('#greeting').text(data.user.role);
        } else {
          alert('Please login');
          window.open('/login.html', '_self');
        }
        console.log(data);        
        const user = data.user; // Assuming only one user in the data
        renderUserInfo(user);
        renderTable(user.order);
      },
      error: function () {
        alert('Please login');
        window.open('/login.html', '_self');
      },
    });
});

function renderUserInfo(user) {
  const userInfoDiv = document.getElementById("userInfo");
  const userInfoHTML = `
    <p><strong>User ID:</strong> ${user.userid}</p>
    <p><strong>Username:</strong> ${user.username}</p>
    <p><strong>Email:</strong> ${user.email}</p>
    <p><strong>Gender:</strong> ${user.gender}</p>
    <p><strong>Birthday:</strong> ${user.birthday}</p>
  `;
  userInfoDiv.innerHTML = userInfoHTML;
}

// Function to render the table
function renderTable(orders) {
  const tableBody = document.getElementById("orderTableBody");
  tableBody.innerHTML = ""; // Clear existing rows

  // Insert rows for each order
  orders.forEach(order => {
    // Parse tickets JSON string into an array
    const tickets = JSON.parse(order.tickets)
      .map(ticket => `${ticket.num} x ${ticket.name} ($${ticket.price})`)
      .join("<br>");

    const row = `
      <tr>
        <td><a href="javascript:void(0)" onclick="goToPage(
       '${order.firstName}',
       '${order.lastName}',
       '${order.email}',
       '${order.address}',
       '${order.region}',
       '${order.district}',
       '${order.paymentMethod}',
       '${order.order_id}', 
       '${order.Purchase_Date}', 
       '${order.Event_Name}', 
       '${order.Event_Date}', 
       '${order.seat_list}', 
       '${tickets}', 
       '${order.total_cost}',

   )">${order.order_id}</a></td>
        <td>${order.Purchase_Date}</td>
        <td>${order.Event_Name}</td>
        <td>${order.Event_Date}</td>
        <td>${order.seat_list}</td>
        <td>${tickets}</td>
        <td>${order.total_cost}</td>
      </tr>
    `;
    tableBody.insertAdjacentHTML("beforeend", row);
  });

  // Update record count
  document.getElementById("recordCount").textContent = orders.length;
}
function goToPage(firstName,lastName,email,address,region,district,paymentMethod,orderid,Purchase_Date,Event_Name,Event_Date,seat_list,tickets,total_cost) {
  // 动态生成目标 URL
  const url = `./after_payment.html`;
  var formData = {
    orderid:orderid,
    firstName: firstName,
    lastName:lastName,
    email:email,
    address:address,
    region:region,
    district:district,
    paymentMethod:paymentMethod,
    moviename: Event_Name,
    moviedate: Event_Date,
    moviecost: total_cost,
    movieseat: seat_list,
};

// Store data in sessionStorage
sessionStorage.setItem('paymentData', JSON.stringify(formData));
var paymentData_movie = {
  orderid:orderid,
  firstName: firstName,
  lastName:lastName,
  email:email,
  address:address,
  region:region,
  district:district,
  paymentMethod:paymentMethod,
  moviename: Event_Name,
  moviedate: Event_Date,
  moviecost: total_cost,
  movieseat: seat_list,
};
sessionStorage.setItem('paymentData_movie', JSON.stringify(paymentData_movie));
  // 跳转到目标页面
  window.location.href = url;
}