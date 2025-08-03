$(document).ready(function () {
  // <!-- XU Zhuoning liu xinyan 21101256d;22097739d -->
  $.ajax({
    url: '/auth/me',
    type: 'GET',
    success: function (data) {
      if (data.status === 'success') {
        $('#username').text(data.user.userid);
        $('#greeting').text(data.user.role);
      } else {
        alert('Please login');
        window.open('/login.html', '_self');
      }
      $("#userid").text(data.user.userid);
      $("#nickname").val(data.user.username);
      $("#password").val(data.user.password);
      $("#staticEmail").text(data.user.email);
      $("#birthday").attr("value",data.user.birthday);;
      $("input[name='gender']").each(function(){
        if($(this).attr("id")==data.user.gender)
          $(this).attr('checked', 'true');;
      })

    },
    error: function () {
      alert('Please login');
      window.open('/login.html', '_self');
    },
  });
  $('#logout').on('click', function () {
    var confirm1 = confirm('Confirm to logout?');
    if (confirm1) {
      $.ajax({
        url: '/auth/logout',
        type: 'GET',
        success: function (data) {
          window.open('/login.html', '_self');
        },
        error: function () {
          alert('Please login');
          window.open('/login.html', '_self');
        },
      });
    }
  });
  $('#update').on('click', function () {
    if ($('#nickname').val() == '' || $('#password').val() == '') {
      alert('Username and password cannot be empty');
    } 
    else {
      var formData = new FormData();
      sessionStorage.setItem("userid",`${$('#userid').attr("placeholder")}`)
      formData.append('userid', `${$('#userid').attr("placeholder")}`);
      formData.append('username', `${$('#nickname').val()}`);
      formData.append('password', `${$('#password').val()}`);
      formData.append('staticEmail', `${$('#staticEmail').val()}`);
      formData.append('birthday', `${$('#birthday').val()}`);
      $("input[name='gender']").each(function(){
        if($(this).prop('checked'))
          formData.append('gender', `${$(this).attr("id")}`);
      })
      var fileInput = $('#formFile')[0]; // 获取文件输入元素
      if (fileInput.files.length > 0) {
        formData.append('profileImage', fileInput.files[0]); // 添加图片文件到 FormData
      }
      for (let pair of formData.entries()) {
        console.log(pair[0] + ': ' + pair[1]);
      }
      $.ajax({
        url: '/auth/update',
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
    }
  });
});
