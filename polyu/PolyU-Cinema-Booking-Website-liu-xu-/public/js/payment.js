var list_movie_name;
var list_movie_date;
var list_movie_cost;
var list_movie_seat;
var formdata;
var id;
var tickets;
// <!-- XU Zhuoning liu xinyan 21101256d;22097739d -->
$(document).ready(function () {
    $.ajax({
        url: '/pay/payment_page',
        type: 'GET',
        success: function (data) {
            if (data.status === 'success') {
                console.log(data);
                data = data.payment_info;
                id=data.Event_id;
                tickets=data.tickets;
                var innerhtml = `
                <div class="col-4">
                    <img src="${data.movie_imagel}" class="w-75 h-100 border rounded" style="object-fit:cover;aspect-ratio: 9 / 14;" alt="${data.movie_name}" style="height:100%">
                </div>
                <div class="col-8">
                    <h5 class="card-title fw-bold my-4">${data.movie_name}</h5>         
                    <h6 class="card-text mb-4"><span class="fw-bold">Showing date: </span>${data.time_select}</h6>
                    <h6 class="card-text mb-4"><span class="fw-bold">Seats booked: </span>${data.seat_list}</h6>
                    <h6 class="card-text mb-4"><span class="fw-bold">Total (HKD)$ </span> ${data.total_cost}</h6>                  
                </div>
                    `;

                list_movie_name = data.movie_name;
                list_movie_date = data.time_select;
                list_movie_seat = data.seat_list;
                list_movie_cost = data.total_cost;
                $('#film_info').append(innerhtml);
                var item_num = 0;
                for (let i = 0; i < data.seat_list.length; i++) {
                    if (data.seat_list[i] === ',') {
                        item_num++;
                    }
                }
                $("#items_num").html(item_num + 1);
                $(JSON.parse(data.tickets)).each(function () {
                    console.log(this);
                    if (this.num == 0) return;
                    innerhtml = `
                        <li class="list-group-item d-flex justify-content-between lh-sm">
                            <div>
                                <h6 class="my-0">${this.name}</h6>
                                <small class="text-body-secondary">${this.num} tickets</small>
                            </div>
                            <span class="text-body-secondary">$${this.price}</span>
                        </li>
                    `;
                    $("#items_list").append(innerhtml);
                });
                innerhtml = `
                        <li class="list-group-item d-flex justify-content-between">
                            <span>Total (HKD)</span>
                            <strong>$${data.total_cost}</strong>
                        </li>
                `;
                $("#items_list").append(innerhtml);

            }
        },

        error: function () {
            alert('Please login');
            window.open('/login.html', '_self');
        },
    })

})
const districtsData = {
    hong_kong_island: [
        { value: "central_and_western", text: "Central and Western" },
        { value: "eastern", text: "Eastern" },
        { value: "southern", text: "Southern" },
        { value: "wan_chai", text: "Wan Chai" }
    ],
    kowloon: [
        { value: "kowloon_city", text: "Kowloon City" },
        { value: "kwun_tong", text: "Kwun Tong" },
        { value: "sham_shui_po", text: "Sham Shui Po" },
        { value: "wong_tai_sin", text: "Wong Tai Sin" },
        { value: "yau_tsim_mong", text: "Yau Tsim Mong" }
    ],
    new_territories: [
        { value: "islands", text: "Islands" },
        { value: "kwai_tsing", text: "Kwai Tsing" },
        { value: "north", text: "North" },
        { value: "sai_kung", text: "Sai Kung" },
        { value: "sha_tin", text: "Sha Tin" },
        { value: "tai_po", text: "Tai Po" },
        { value: "tsuen_wan", text: "Tsuen Wan" },
        { value: "tuen_mun", text: "Tuen Mun" },
        { value: "yuen_long", text: "Yuen Long" }
    ]
};

function updateDistricts() {
    const regionSelect = document.getElementById('region');
    const districtSelect = document.getElementById('district');
    const selectedRegion = regionSelect.value;

    // Clear previous options
    districtSelect.innerHTML = '<option value="">Select a district</option>';

    if (selectedRegion && districtsData[selectedRegion]) {
        districtsData[selectedRegion].forEach(district => {
            const option = document.createElement('option');
            option.value = district.value;
            option.textContent = district.text;
            districtSelect.appendChild(option);
        });
    }
}

document.addEventListener('DOMContentLoaded', function () {
    const creditRadio = document.getElementById('credit');
    const paypalRadio = document.getElementById('paypal');
    const creditCardSection = document.getElementById('credit_card');
    const paypalSection = document.getElementById('paypal_info');

    function togglePaymentSections() {
        if (creditRadio.checked) {
            creditCardSection.classList.remove('d-none');
            paypalSection.classList.add('d-none');
        } else if (paypalRadio.checked) {
            creditCardSection.classList.add('d-none');
            paypalSection.classList.remove('d-none');
        }
    }

    creditRadio.addEventListener('change', togglePaymentSections);
    paypalRadio.addEventListener('change', togglePaymentSections);

    togglePaymentSections(); // Initialize on page load
});

function submitCheckoutFormmovie() {

    // Gather form data
    var formData_movie = {
        moviename: list_movie_name,
        moviedate: list_movie_date,
        moviecost: list_movie_cost,
        movieseat: list_movie_seat,
    };

    // Store data in sessionStorage
    sessionStorage.setItem('paymentData_movie', JSON.stringify(formData_movie));
}
function generateOrderId(userId) {
    const timestamp = Date.now(); // 获取当前时间戳
    return `${timestamp}_${userId}`; // 拼接时间戳和用户 ID
}

function submitCheckoutForm(event) {
    event.preventDefault(); // Prevent normal form submission
    let isValid = true; // 初始化标记为表单有效

    // 遍历需要验证的输入框
    $('#cc-number, #cc-expiration, #cc-cvv').each(function () {
      if ($(this).val().trim() === '') {
        isValid = false; // 如果有空值，标记为无效
        console.error($(this).attr('id') + ' is empty.'); // 打印错误到控制台
      }
    });

    // 如果验证未通过，弹出提示
    if (!isValid) {
      alert('Please fill in all required fields.');
      return;
    } else {
      alert('Payment submitted successfully!');
      // 这里可以执行实际的提交逻辑，例如 AJAX 请求
    }
    var username=sessionStorage.getItem('username');
    var userid=sessionStorage.getItem('userid');
    var order_id=generateOrderId(username);
    // Gather form data
    var formData = {
        order_id:order_id,
        firstName: document.getElementById('firstName').value,
        lastName: document.getElementById('lastName').value,
        email: document.getElementById('email').value,
        address: document.getElementById('address').value,
        region: document.getElementById('region').value,
        district: document.getElementById('district').value,
        paymentMethod: document.querySelector('input[name="paymentMethod"]:checked').value,
        // Additional fields can be added here
    };

    // Store data in sessionStorage
    sessionStorage.setItem('paymentData', JSON.stringify(formData));
    submitCheckoutFormmovie();
    // Redirect to the payment success page




    var formDataObject=formData;

    var Purchase_Date=new Date();
    var event_id=id;
    var Event_Name=list_movie_name;
    var Event_Date=list_movie_date;
    var total_cost=list_movie_cost;
    var seat_list=list_movie_seat;    
    formData = new FormData();
    for (var key in formDataObject) {
        formData.append(key, formDataObject[key]);
      }     
    formData.append('username', username);
    formData.append('order_id', order_id);
    formData.append('Purchase_Date', Purchase_Date);
    formData.append('Event_id',event_id);
    formData.append('Event_Name', Event_Name);
    formData.append('Event_Date', Event_Date);
    formData.append('total_cost', total_cost);
    formData.append('seat_list',seat_list);
    formData.append('tickets',tickets);
    $.ajax({
        url: '/pay/order',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function (data) {
            if (data.status === 'success') {
                alert(`Update successfully}!`);
                window.open('/login.html', '_self');
            } else {
                alert(data.message);
            }
        },
        error: function (xhr) {
            const response = xhr.responseJSON || JSON.parse(xhr.responseText);
            if (response && response.message) {
                alert(response.message);
            } else {
                alert('Unknown error');
            }
            console.error('Error:', response);
        },
    });
    window.location.href = "after_payment.html";

}