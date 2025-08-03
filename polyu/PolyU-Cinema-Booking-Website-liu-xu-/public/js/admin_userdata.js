var userData;
// <!-- XU Zhuoning liu xinyan 21101256d;22097739d -->
$(document).ready(function () {
    $.ajax({
      url: '/admin/all_order',
      type: 'GET',
      success: function (data) {
        if (data.status === 'success') {
          $('#username').text(data.user.username);
          $('#greeting').text(data.user.role);
            userData=data.user;
            console.log(userData.user);
        }      
        const user = data.user; // Assuming only one user in the data
        renderUserInfo(user);
        displayUsers();
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

  `;
  userInfoDiv.innerHTML = userInfoHTML;
}

// Function to render the table
function displayUsers() {
    console.log($('#user-table'));
        const tableBody = document.getElementById('user-table');

        userData.user.forEach(user => {
            const row = document.createElement('tr');
            const orderCount = user.order ? user.order.length : 0; // Count the number of orders

            row.innerHTML = `
                <td>${user.userid}</td>
                <td>${user.username}</td>
                <td>${user.email}</td>
                <td>${user.birthday}</td>
                <td>${user.gender}</td>
                <td>${user.role}</td>
                <td>${user.enabled}</td>
                <td>${orderCount}</td>
            `;
            tableBody.appendChild(row);
        });
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

