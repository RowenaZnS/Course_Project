/* eslint-disable no-useless-escape */
// <!-- XU Zhuoning liu xinyan 21101256d;22097739d -->
$(document).ready(function () {
  $('#role').onchange(function () {
    // Get the selected value
    const selectedValue = $(this).val();

    // Check if the value is "admin"
    if (selectedValue === 'admin') {
      // Show an alert
      alert('You cannot select "admin".');

      // Change the value back to "user"
      $(this).val('user');
    }
  });
    $('#email').on('change',function(email_temp,id){
      var email=$('#email').val();
      var reg = /^[A-Za-z0-9]+([_\.][A-Za-z0-9]+)*@([A-Za-z0-9\-]+\.)+[A-Za-z]{2,6}$/;
      if(!reg.test(email)){
          $('#emailinfo').html("Please enter correct E-mail format like example@quiz.com");
          $("#emailinfo").css("color","red");
      }
      else
        $('#emailinfo').html("");
  });
    $('#register').on('click', function () {
      if ($('#nickname').val() == '' || $('#password').val() == '') {
        alert('Username and password cannot be empty');
      } 
      else if($('#password').val() !=$('#repassword').val()){
        alert('Password mismatch!');
      }
      else if($('#role').val() =="PleaseSelect"){
        alert('Please select your role');
      }
      else {
        var formData = new FormData();
        formData.append('userid', `${$('#userid').val()}`);
        alert($('#userid').val());
        formData.append('username', `${$('#nickname').val()}`);
        formData.append('password', `${$('#password').val()}`);
        formData.append('role', `${$('#role').val()}`);
        formData.append('enabled', true);
        formData.append('birthday', `${$('#birthday').val()}`);
        formData.append('email', `${$('#email').val()}`);
        $.ajax({
          url: '/auth/register',
          type: 'POST',
          data: formData,
          processData: false,
          contentType: false,
          success: function (data) {
            if (data.status === 'success') {
              alert(`Welcome, ${$('#nickname').val()}! \n You can login with your account now!`);
              window.location.href = '/login.html';
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
      }
    });
    $('#back').on('click', function () {
      window.location.href = '/login.html';
    });
  });
  