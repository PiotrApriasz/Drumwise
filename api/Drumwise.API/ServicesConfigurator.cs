using System.Net;
using System.Reflection;
using Ardalis.GuardClauses;
using Drumwise.Application.Common.Behaviours;
using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Common.Mappings;
using Drumwise.Application.Common.Models.Settings;
using Drumwise.Application.Services;
using Drumwise.Application.Services.Files;
using Drumwise.Application.Services.Mailing;
using Drumwise.Features.Homeworks;
using Drumwise.Features.Homeworks.Validation;
using Drumwise.Infrastructure.Data;
using Drumwise.Infrastructure.Data.Files;
using Drumwise.Infrastructure.Data.Interceptors;
using Drumwise.Infrastructure.Identity;
using Drumwise.Infrastructure.Identity.Constants;
using Drumwise.Infrastructure.Identity.Validators;
using FluentValidation;
using FluentValidation.Results;
using MediatR;
using MediatR.Pipeline;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Diagnostics;
using SharpGrip.FluentValidation.AutoValidation.Endpoints.Extensions;
using SharpGrip.FluentValidation.AutoValidation.Endpoints.Results;
using SharpGrip.FluentValidation.AutoValidation.Shared.Extensions;

namespace Drumwise.API;

public static class ServicesConfigurator
{
    public static IServiceCollection ConfigureIdentity(this IServiceCollection services, IConfiguration configuration)
    {
        var connectionString = configuration.GetConnectionString("IdentityConnection");
        //Guard.Against.Null(connectionString, $"Connection string for 'IdentityConnection' not found");
        
        services.AddScoped<IUser, CurrentUser>();
        services.AddHttpContextAccessor();
        
        services.AddDbContext<AppIdentityDbContext>((sp, options) =>
        {
            options.UseSqlite("DataSource=app.db");
        });
        
        services.AddScoped<IApplicationDbContext>(provider => provider.GetRequiredService<AppIdentityDbContext>());
        services.AddScoped<AppIdentityDbContextInitializer>();

        services.AddIdentityApiEndpoints<ApplicationUser>(options =>
            {
                options.SignIn.RequireConfirmedEmail = false;
            })
            .AddRoles<IdentityRole>()
            .AddEntityFrameworkStores<AppIdentityDbContext>();

        services.AddAuthorization(options =>
        {
            options.AddPolicy(Policies.CanAddHomework, policy => policy.RequireRole(Roles.Teacher));
            options.AddPolicy(Policies.RequireTeacherRole, policy => policy.RequireRole(Roles.Teacher));
        });
        
        services.AddValidatorsFromAssemblyContaining<AdditionalUserDataRequestValidator>();
        
        services.AddTransient<IIdentityService, IdentityService>();
        services.AddTransient<IEmailSender<ApplicationUser>, IdentityEmailSender>();

        return services;
    }
    
    public static IServiceCollection AddInfrastructuresServices(this IServiceCollection services, IConfiguration configuration)
    {
        var connectionString = configuration.GetConnectionString("AppConnection");
        //Guard.Against.Null(connectionString, $"Connection string for 'AppConnection' not found");

        services.AddScoped<ISaveChangesInterceptor, AuditableEntityInterceptor>();
        services.AddScoped<ISaveChangesInterceptor, DispatchDomainEventsInterceptor>();

        services.AddDbContext<ApplicationDbContext>((sp, options) =>
        {
            options.AddInterceptors(sp.GetServices<ISaveChangesInterceptor>()!);
            options.UseSqlite("DataSource=main.db");
        });
        
        services.AddValidatorsFromAssemblyContaining<CreateHomeworkCommandValidator>();

        services.AddScoped<IApplicationDbContext>(provider => provider.GetRequiredService<ApplicationDbContext>());
        services.AddScoped<ApplicationDbContextInitializer>();
        
        var googleDriveApiSettings = configuration.GetSection("GoogleDriveApiSettings").Get<GoogleDriveApiSettings>();
        services.AddSingleton(googleDriveApiSettings!);
        services.AddTransient<IFileStorageService, GoogleDriveService>();

        return services;
    }

    public static IServiceCollection AddApplicationServices(this IServiceCollection services, IConfiguration configuration)
    {
        var typeWithMapFromInterface = typeof(HomeworkItemBriefDto);

        services.AddAutoMapper(cfg =>
        {
            cfg.AddProfile<MappingProfile>();
        }, typeWithMapFromInterface.Assembly);

        services.AddMediatR(cfg =>
        {
            cfg.RegisterServicesFromAssemblyContaining<CreateHomeworkHandler>();
            cfg.AddBehavior(typeof(IPipelineBehavior<,>), typeof(UnhandledExceptionBehaviour<,>));
            cfg.AddBehavior(typeof(IPipelineBehavior<,>), typeof(PerformanceBahaviour<,>));
            cfg.AddBehavior(typeof(IRequestPreProcessor<>), typeof(LoggingBehaviour<>));
        });
        
        services.AddFluentValidationAutoValidation(config =>
        {
            config.OverrideDefaultResultFactoryWith<CustomValidationResultFactory>();
        });

        var smtpSettings = configuration.GetSection("SmtpSettings").Get<SmtpSettings>();
        services.AddTransient<IFileService, FileService>();
        services.AddTransient<IMailSender, MailSender>();
        services.AddSingleton(smtpSettings!);
        services.AddTransient<IConversionService, ConversionService>();

        return services;
    }
    
    private class CustomValidationResultFactory : IFluentValidationAutoValidationResultFactory
    {
        public IResult CreateResult(EndpointFilterInvocationContext context, ValidationResult validationResult)
        {
            var validationProblemDetails = new ValidationProblemDetails()
            {
                Type = "https://tools.ietf.org/html/rfc9110#section-15.5.1",
                Status = StatusCodes.Status400BadRequest,
                Title = "One or more validation errors occurred."
            };
            
            foreach (var error in validationResult.Errors)
            {
                validationProblemDetails.Errors.Add(error.ErrorCode, [error.ErrorMessage]);
            }

            return Results.BadRequest(validationProblemDetails);
        }
    }
}