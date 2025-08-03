$(document).ready(function () {
  // <!-- XU Zhuoning liu xinyan 21101256d;22097739d -->
  window.onload = function() {
    const rememberedUser = localStorage.getItem('username');
    const rememberedPass = localStorage.getItem('password');
    if (rememberedUser) {
        document.getElementById('username').value = rememberedUser;
    }
    if (rememberedPass) {
        document.getElementById('password').value = rememberedPass;
    }
};

// 表单提交时，处理“记住我”功能
document.getElementById('loginForm').onsubmit = function() {
    const rememberMe = document.getElementById('rememberMe').checked;
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    if (rememberMe) {
        localStorage.setItem('username', username);
        localStorage.setItem('password', password);
    } else {
        localStorage.removeItem('username');
        localStorage.removeItem('password');
    }
};
  $('#login').on('click', function () {
    if ($('#username').val() == '' || $('#password').val() == '') {
      alert('Username and password cannot be empty');
    } else {
      var formData = new FormData();
      formData.append('username', `${$('#username').val()}`);
      formData.append('password', `${$('#password').val()}`);
      $.ajax({
        url: '/auth/login',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function (data) {
          if (data.status === 'success') {
            if(data.user.role=='admin'){
              alert("admin");
              window.location.href = '/admin_show_movie.html';
            }

            if(data.user.role=='user'){
              sessionStorage.setItem("username",`${data.user.username}`)
              alert(`Logged as \`${data.user.username}\` `);
              window.location.href = '/myprofile.html';
            }

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
  $('#register').on('click', function () {
    window.location.href = '/register.html';
  });
});
