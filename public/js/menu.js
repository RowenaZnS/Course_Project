$(document).ready(function () {
    // <!-- XU Zhuoning liu xinyan 21101256d;22097739d -->
    $.get("./assets/drink-menu.json", function (data) {
        console.log(data);
        let drink_data=JSON.parse(window.JSON.stringify(data));
        $(data).each(function(){
        let innerhtml=`
        <div class="col mb-3">
            <div class="card  h-100">
            <div class="w-100">
                <img src="${$(this)[0].image}" class="w-100 border rounded" style="object-fit:cover;aspect-ratio: 9 / 14;" alt="${$(this)[0].name}">
            </div>
                <div class="card-body">
                    <h6 class="card-title fw-bold">${$(this)[0].name}</h6>
                    <h4 class="badge text-bg-success">${$(this)[0].type}</h4>
                    <p class="card-text">${$(this)[0].price}</p>
                </div>
            </div>
        </div>`;
        $("#drink-menu").append(innerhtml);
        });
    }).fail(function (error) {
         let innerhtml=`
         <div class="col mb-3">
            <div class="alert alert-danger" role="alert">
                Failed to fetch drink menu. Please try again later.
            </div>
        </div>`;
            $("#menu_container").append(innerhtml);
    });
});