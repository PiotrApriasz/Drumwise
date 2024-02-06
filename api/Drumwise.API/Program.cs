using Drumwise.API;
using Drumwise.API.Endpoints;
using Drumwise.API.ExceptionHandlers;
using Drumwise.Infrastructure.Data;
using Drumwise.Infrastructure.Identity;
using NLog;
using NLog.Web;
using SharpGrip.FluentValidation.AutoValidation.Endpoints.Extensions;

var logger = LogManager.Setup().LoadConfigurationFromAppSettings().GetCurrentClassLogger();
logger.Debug("Api init");

try
{
    var builder = WebApplication.CreateBuilder(args);
    
    builder.Logging.ClearProviders();
    builder.Host.UseNLog();

    // Add services to the container.
    // Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
    builder.Services.AddEndpointsApiExplorer();
    builder.Services.AddSwaggerGen();

    builder.Services.AddExceptionHandler<BadHttpRequestExceptionHandler>();
    builder.Services.AddExceptionHandler<ServerExceptionHandler>();
    builder.Services.AddProblemDetails();

    builder.Services.ConfigureIdentity(builder.Configuration);
    builder.Services.AddApplicationServices(builder.Configuration);

    var app = builder.Build();

    // Configure the HTTP request pipeline.
    if (app.Environment.IsDevelopment())
    {
        app.UseSwagger();
        app.UseSwaggerUI();

        await app.InitializeIdentityDatabaseAsync();
    }

    app.UseHttpsRedirection();

    app.UseExceptionHandler();

    // Map endpoints --------------------------------------------------
    var apiEndpoints = app.MapGroup("/api").AddFluentValidationAutoValidation();

    apiEndpoints.MapIdentityApi<ApplicationUser>();
    apiEndpoints.MapAdditionalIdentityEndpoints();
    // ----------------------------------------------------------------

    app.Run();

    throw new Exception("DUUUPPAPAPAP");
}
catch(Exception exception)
{
    logger.Error(exception, "Stopped program because of exception");
    throw;
}
finally
{
    LogManager.Shutdown();
}
